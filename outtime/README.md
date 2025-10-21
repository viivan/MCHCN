<p float="left"><img src="https://img.shields.io/badge/python-v3.7-red"> <img src="https://img.shields.io/badge/tensorflow-v1.14-blue"> <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Coder-Yu/QRec"></p>
<h2>Introduction</h2>

**MCHCN** is a Python framework for service recommendation systems (Supported by Python 3.7.4 and Tensorflow 1.14+)

<h2>Architecture</h2>

![MCHCN Architecture](../Overview%20of%20the%20proposed%20MCHCN%20method.png)

<h2>Features</h2>
<ul>
<li><b>1. </b>: By studying the calling relationship between mashups and APIs, in addition to the tag information of mashups, we define four types of motifs and construct the corresponding hypergraphs. The constructed hypergraphs not only extract effective information but also significantly address the issue of sparse data in the calling relationships between mashups and APIs.</li>
<li><b>2. </b>We propose a hybrid random walk approach that retains the advantages of partially absorbing random walks and biased random walks. The approach captures global information, and balances homophily and structural equivalence. We adopt the approach to optimize the hypergraph structure and guide the subsequent hypergraph convolution process.</li>
<li><b>3. </b>To further improve the accuracy of the model, we incorporate a channel error attention mechanism to help to correctly allocate weights when channels are aggregated, and use a contrastive learning method to reduce noise introduced by a large number of aggregation operations. We verified the effectiveness of the MCHCN method in service recommendation scenarios using experiments on real mashup calling datasets.</li>
</ul>
<h2>Requirements</h2>
<ul>
<li>gensim==4.1.2</li>
<li>joblib==1.1.0</li>
<li>mkl==2022.0.0</li>
<li>mkl_service==2.4.0</li>
<li>networkx==2.6.2</li>
<li>numba==0.53.1</li>
<li>numpy==1.20.3</li>
<li>scipy==1.6.2</li>
<li>tensorflow==1.14.0</li>
</ul>
<h2>Usage</h2>
<p>There the way to run the recommendation models in MCHCN:</p>
<ul>
<li>1.Configure the xx.conf file in the directory named config. (xx is the name of the model you want to run)</li>
<li>2.Run main.py.</li>
</ul>


<h2>Related Datasets</h2>

[//]: # ()
[//]: # (<h3>Reference</h3>)

[//]: # (<p>[1]. Tang, J., Gao, H., Liu, H.: mtrust:discerning multi-faceted trust in a connected world. In: International Conference on Web Search and Web Data Mining, WSDM 2012, Seattle, Wa, Usa, February. pp. 93–102 &#40;2012&#41;</p>)

[//]: # (<p>[2]. Massa, P., Avesani, P.: Trust-aware recommender systems. In: Proceedings of the 2007 ACM conference on Recommender systems. pp. 17–24. ACM &#40;2007&#41; </p>)

[//]: # (<p>[3]. G. Zhao, X. Qian, and X. Xie, “User-service rating prediction by exploring social users’ rating behaviors,” IEEE Transactions on Multimedia, vol. 18, no. 3, pp. 496–506, 2016.</p>)

[//]: # (<p>[4]. Iván Cantador, Peter Brusilovsky, and Tsvi Kuflik. 2011. 2nd Workshop on Information Heterogeneity and Fusion in Recom- mender Systems &#40;HetRec 2011&#41;. In Proceedings of the 5th ACM conference on Recommender systems &#40;RecSys 2011&#41;. ACM, New York, NY, USA</p>)

[//]: # (<p>[5]. Yu et al. Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation, WWW'21.</p>)
[//]: # (<p>[6]. He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, SIGIR'20.</p>)
<h2>Acknowledgment</h2>
<p>This work is supported by the “Pioneer” and “Leading Goose” R&D Program of Zhejiang Province, China (No. 2025C01022, 2023C01022), the LingYan Planning Project of Zhejiang Province, China (No. 2023C01215),the Science and Technology Key Research Planning Project of HuZhou City,China (NO. 2022ZD2019) and Research of Key Technologies for Baishanzu National Park of China (NO. 2022JBGS01).</p>

[//]: # (运行环境、核心文件、入口)