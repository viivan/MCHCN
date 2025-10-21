from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
from util.loss import bpr_loss
import os
from util import config
from math import sqrt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class MCHCN(SocialRecommender, GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation, fold=fold)
        # Load mashup tag data
        self.mashup_tag = self.load_mashup_tag(conf['MCHCN']['-tag_path'])
        self.num_tags = len(set(tag for tags in self.mashup_tag.values() for tag in tags))
        self.R = self.build_mashup_tag_matrix()

    def load_mashup_tag(self, tag_path):
        "Load mashup tag data (path read from MCHCN.conf)"
        mashup_tag = {}
        with open(tag_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                mid, tid = line.strip().split(',')
                if mid not in self.data.user:
                    continue
                mid_idx = self.data.user[mid]
                if mid_idx not in mashup_tag:
                    mashup_tag[mid_idx] = []
                mashup_tag[mid_idx].append(int(tid))
        return mashup_tag

    def build_mashup_tag_matrix(self):
        """ Construction of the mashup-tag binary matrix R: N×J """
        N = self.num_users  # 确保user对应mashup
        J = self.num_tags
        R = np.zeros((N, J), dtype=np.float32)
        for mid_idx, tids in self.mashup_tag.items():
            for tid in tids:
                R[mid_idx, tid] = 1.0
        return R

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_users), dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        """Construct the mashup-API call matrix T: N× A:"""
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return ratingMatrix

    def buildJointAdjacency(self):
        """ Build a Normalized mashup-API Adjacency matrix """
        indices = [[self.data.user[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users, self.num_items])
        return norm_adj

    def compute_hybrid_random_walk_weight(self, S, delta=0.4, p=4, q=1):
        """Section 3.4.1 of the Paper: Calculating Convolutional Weight Matrix C with Hybrid Random Walk"""
        N = S.shape[0]
        S_dense = S.toarray()
        d = np.sum(S_dense, axis=1).reshape(-1, 1)
        d[d == 0] = 1e-6

        # Calculate the shortest distance matrix d_ij
        d_ij = np.full((N, N), float('inf'))
        d_ij[S_dense > 0] = 1.0
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if d_ij[i, j] > d_ij[i, k] + d_ij[k, j]:
                        d_ij[i, j] = d_ij[i, k] + d_ij[k, j]
        d_ij[d_ij == float('inf')] = 2.0

        # Build a Q matrix
        Q = np.ones((N, N), dtype=np.float32)
        Q[d_ij == 0] = 1.0 / p
        Q[d_ij == 2] = 1.0 / q

        # Calculate the L and C matrices
        W = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j and d[j] > 0:
                    W[i, j] = S_dense[i, j] / d[j]
        D = np.diag(np.sum(W * Q, axis=1))
        L = D - W * Q
        Lambda = delta * np.eye(N, dtype=np.float32)
        C = np.linalg.inv(Lambda + L) @ Lambda

        return coo_matrix(C)

    def buildMotifInducedAdjacencyMatrix(self):
        """Section 3.3 of the Paper: Constructing Hypergraph Adjacency Matrices S1-S4 Based on 4 Types of Motifs"""
        N = self.num_users
        T = self.buildSparseRatingMatrix().toarray()
        R = self.R

        # 1. Cross Correlated Motif (S1)
        R_hat = R @ R.T
        R_star = np.triu((R_hat > 0).astype(float), k=1)
        S1 = (R_star @ R_star) * R_star
        S1 = S1 + S1.T

        # 2. Multiple Strongly Correlated Motif (S2)
        R_prime = np.triu((R_hat >= 2).astype(float), k=1)
        S2 = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                count = np.sum(R_prime[:, i] * R_prime[:, j])
                S2[i, j] = 1.0 if count > 1 else 0.0
        S2 = S2 + S2.T

        # 3. Single Relation Motif (S3)
        T_hat = T @ T.T
        T_star = np.triu((T_hat > 0).astype(float), k=1)
        S3 = T_star @ T_star.T
        S3 = S3 + S3.T

        # 4. Multi-relational Motif (S4)
        S4 = (T_star @ T_star.T) * R_star
        S4 = S4 + S4.T

        # Row Normalization
        def normalize_adj(adj):
            adj = adj + np.eye(N)
            degree = np.sum(adj, axis=1).reshape(-1, 1)
            return adj / degree

        S1 = normalize_adj(S1)
        S2 = normalize_adj(S2)
        S3 = normalize_adj(S3)
        S4 = normalize_adj(S4)

        return [coo_matrix(S1), coo_matrix(S2), coo_matrix(S3), coo_matrix(S4)]

    def adj_to_sparse_tensor(self, adj):
        """Sparse Matrix to TensorFlow Sparse Tensor"""
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj

    def dual_scale_contrastive_loss(self, Pm, Pm_tilde, gamma=0.1):
        """Section 3.5 of the Paper: Two-scale Contrastive Learning Loss (InfoNCE)"""
        N = tf.shape(Pm)[0]
        sim_matrix = tf.matmul(Pm, tf.transpose(Pm_tilde)) / gamma
        labels = tf.range(N, dtype=tf.int32)
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim_matrix)
        return tf.reduce_mean(ce_loss)

    def initModel(self):
        """Initialize the model: Adapt to four types of Motif hypergraphs and mixed random walks"""
        super(MCHCN, self).initModel()
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        self.n_channel = 4
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        initializer = tf.contrib.layers.xavier_initializer()

        # Calculate the mixed random walk weight matrix C1-C4
        self.C_matrices = []
        for S in M_matrices:
            C = self.compute_hybrid_random_walk_weight(S, delta=0.4, p=4, q=1)
            self.C_matrices.append(self.adj_to_sparse_tensor(C))

        # Initialize the channel gating and attention parameters
        self.weights = {}
        for i in range(self.n_channel):
            self.weights['gating%d' % (i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i+1)] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')

        # 通道门控函数
        def self_gating(em, channel):
            return tf.multiply(em, tf.nn.sigmoid(tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))
        def channel_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), 1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings, score

        # Initialize the mashup embeddings of each channel
        user_embeddings_list = []
        for i in range(self.n_channel):
            gate_emb = self_gating(self.user_embeddings, i+1)
            user_embeddings_list.append(gate_emb)
        all_embeddings_list = [[emb] for emb in user_embeddings_list]

        # Initialize the API embedding
        item_embeddings = self.item_embeddings
        all_item_embeddings = [item_embeddings]
        R = self.buildJointAdjacency()

        # Multi-channel hypergraph convolution
        self.ss_loss = 0
        for k in range(self.n_layers):
            mixed_mashup_emb = channel_attention(*[emb_list[-1] for emb_list in all_embeddings_list])[0]
            # Convolution of each channel (using mixed random walk weights C)
            for i in range(self.n_channel):
                C = self.C_matrices[i]
                new_emb = tf.sparse_tensor_dense_matmul(C, all_embeddings_list[i][-1])
                new_emb = tf.math.l2_normalize(new_emb, axis=1)
                all_embeddings_list[i].append(new_emb)
            # API convolution
            new_item_emb = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_mashup_emb)
            new_item_emb = tf.math.l2_normalize(new_item_emb, axis=1)
            all_item_embeddings.append(new_item_emb)
            item_embeddings = new_item_emb

        # Integrate the embeddings of each channel
        final_mashup_emb_list = []
        for emb_list in all_embeddings_list:
            final_emb = tf.reduce_sum(emb_list, axis=0)
            final_mashup_emb_list.append(final_emb)
        self.final_user_embeddings, self.attention_score = channel_attention(*final_mashup_emb_list)
        self.final_item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0)

        # Calculate the loss of dual-scale contrastive learning
        Pm_tilde = tf.sparse_tensor_dense_matmul(R, self.final_item_embeddings)
        self.ss_loss = self.dual_scale_contrastive_loss(self.final_user_embeddings, Pm_tilde, gamma=0.1)

        # embedded query
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)

    def readConfiguration(self):
        """Read the MCHCN.conf configuration """
        super(MCHCN, self).readConfiguration()
        args = config.OptionConf(self.config['MCHCN'])  # 读取[MCHCN]节点配置
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])
        self.emb_size = int(args.get('-emb_size', 100))  # 从配置读取嵌入维度，默认100

    def trainModel(self):
        """Training model: Multi-objective loss (BPR+ contrastive loss)"""
        # BPR recommendation loss
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        # L2
        reg_loss = 0
        for key in self.weights:
            reg_loss += 1e-5 * tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        # total loss
        total_loss = rec_loss + reg_loss + self.ss_rate * self.ss_loss

        # optimizer
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Training
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l_rec, l_ss = self.sess.run([train_op, rec_loss, self.ss_loss],
                                             feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo, 'training:', epoch + 1, 'batch', n,
                      'rec loss:', round(l_rec, 4), 'ss loss:', round(l_ss, 4))
            # Evaluate and save the optimal embedding
            self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
            self.ranking_performance(epoch)
        self.U, self.V = self.bestU, self.bestV

    def saveModel(self):
        """Save the optimal embedding"""
        self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])

    def predictForRanking(self, u):
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items