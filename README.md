# RL-HFRx
#auther Yao.Y yyao64-c@my.cityu.edu.hk
RL-HFRx, an expertise-embedded, non-parametric offline RL algorithm, to derive optimal patient-specific medication recommendations from EHR data. 
<img width="416" height="299" alt="image" src="https://github.com/user-attachments/assets/3b1e02fc-2633-43ac-b146-a537247b24b1" />
Supplementary Algorithm RL-HFRx’s pseudocode
Algorithm RL-HFRx 
History representation learning:
Input: Training dataset D_{train}, mini-batch size M
1: Initialize RNN cell R_\theta or neural integral function f_\theta, hidden state h_0, linear output function l_\omega, hidden state buffer of training dataset B_{htr}
2: for t in \left[1,2,\ldots,M\right] do
3:	Sample series of observations and actions o_{0:T},a_{0:T}~ D_{train}
4: 	h_{1:T}=R_\theta\left(h_0,o_{0:T-1},a_{0:T-1}\right)\ or\ h_{1:T}=f_\theta\left(h_0,o_{0:T-1},a_{0:T-1}\right)\ ,\ \widehat{o_{1:T}}=l_\omega(h_{1:T})
5:	Update by \theta\prime,\omega\prime\gets arg{min}_{\theta,\omega}L(o_{1:T},\widehat{o_{1:T}})
6: end for
7: for\ patient\ i in\ D_{train} do
8:	Get series of observations and actions o_{i,0:T},a_{i,0:T}~ D_{train}
9: 	h_{i,\ 1:T}=R_\theta\left(h_0,o_{i,0:T-1},a_{i,0:T-1}\right)\ or\ h_{i,\ 1:T}=f_\theta\left(h_0,o_{i,0:T-1},a_{i,0:T-1}\right)
10: 	B_{htr}\gets B_{htr}\cup h_{i,\ 1:T} 
Offline RL:
Input: Training dataset D_{train} and test dataset D_{test}, mini-batch size N, hidden state buffer of training dataset B_{htr}, target network update rate \tau, discount rate \gamma
11: Initialize double deep Q-network Q_{\theta_1},Q_{\theta_2}, target network  Q_{\theta_1\prime},Q_{\theta_2\prime}\ V-network V_\varphi, evaluation buffer B_e, hidden state buffer of training dataset B_{hte}
12: for t in \left[1,2,\ldots,N\right]do 
13:	Sample mini batch of transition \left(o,a,r,o^\prime\right)~D_{train}, (h,h\prime)~B_{htr}
14:	s=o^h,\  s\prime=o\prime^h\prime
	Q-update: 
15:	Search for clinician past action set A at nearest states of s
16:	V_\varphi(s)={max}_{a\epsilon A}\ \funcapply[\lambda m i n{\left(Q_{\theta_1\prime}\left(s,a\right),Q_{\theta_2\prime}\left(s,a\right)\right)}+\left(1-\lambda\right)max{\left(Q_{\theta_1\prime}\left(s,a\right),Q_{\theta_2\prime}\left(s,a\right)\right)}]
17:	\theta_1,\theta_2\gets{argmin}_{\theta_1,\theta_2}({{(Q}_{\theta_1}\left(s,a\right)-r-\gamma V_\varphi(s\prime))}^2+{{(Q}_{\theta_2}\left(s,a\right)-r-\gamma V_\varphi(s\prime))}^2), 
	\varphi\gets{argmin}_\varphi{(y(s,a)-V_\varphi(s))}^2
18:	Update the target network: {\theta\prime}_1\gets{\tau\theta}_1+(1-\tau){\theta\prime}_1,{\theta\prime}_2\gets{\tau\theta}_2+(1-\tau){\theta\prime}_2
19: end for 
20: for\ patient\ i in D_{test} do
21: 	Get series of observations and actions o_{i,0:T},a_{i,0:T}~D_{test}
22:	h_{i,\ 1:T}=R_\theta\left(h_0,o_{i,0:T-1},a_{i,0:T-1}\right)\ or\ h_{i,\ 1:T}=f_\theta\left(h_0,o_{i,0:T-1},a_{i,0:T-1}\right)
23:	s_{i,0:T-1}=o_{i,0:T-1}^h_{i,0:T-1}, {s\prime}_{i,1:T},=o_{i,1:T}^h_{i,1:T}
24:	Search for clinician past action set A_{0:T-1}\prime at nearest states of s_{i,0:T-1}, A_{1:T}\prime at nearest states of {s\prime}_{i,1:T}
25:	a_{i,0:T-1}^\ast={argmax}_{a\epsilon A_{0:T-1}\prime}\funcapply[Q_{\theta_1\prime}\left(s_{i,0:T-1},a\right)],
 a_{i,1:T}^\ast={argmax}_{a\epsilon A_{1:T}\prime}\funcapply[Q_{\theta_1\prime}\left({s\prime}_{i,1:T},a\right)]	
26: 	B_e\gets B_e\cup(a_{i,0:T-1}^\ast,a_{i,1:T}^\ast)
27:	B_{hte}\gets B_{hte}\cup h_{i,\ 1:T}
Fitted Q-evaluation: 
Input: Test dataset D_{test}, evaluation buffer B_e, hidden state buffer of training dataset B_{hte}, mini-batch size N‘, discount rate \gamma, target network update rate \tau
28: Initialize Q-network Q_\theta, target network Q_{\theta\prime}
29: for t in \left[1,2,\ldots,N\prime\right]do
30:	Sample mini batch of transition \left(o,a,r,o^\prime\right)~D_{test}, (h,h\prime)~B_{hte}, (a\ast,a\ast\prime)~B_e
31:	s=o^h,\  s\prime=o\prime^h\prime
32:	\theta\gets{argmin}\below{\theta\prime}{(Q_{\theta\prime}(s,a)-(r+\gamma Q_{\theta\prime}(s\prime,a\ast\prime))}
33:	Update the target network: \theta\prime\gets\tau\theta+(1-\tau)\theta
34: end for
35: Output: \widehat{C(}s)=Q_\theta(s,a),\forall s~D_{test}


<img width="415" height="686" alt="image" src="https://github.com/user-attachments/assets/3e175c63-9c19-400c-93d1-00c640b47675" />
