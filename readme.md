## Physics-Informed Extreme Learning Machine

### (1) Differential Equation

Lorentz Force

$$ \vec{F} = q \vec E + \vec v \times \vec B  $$

Newton Law of Motion

$$ \vec F = m \vec a $$

Equating we have

$$ m \frac{d^2}{dt^2} \vec r(t) = q \vec E + \frac{d}{dt} \vec r(t) \times \vec B $$

### (2) Applying Initial Conditions

Applying Initial Conditions as per Physics Informed Extreme Learning Method known as Ansatz Constraint

Let

$$ \vec r(t) = \vec r_0 \ + \vec {v_0}t \ + \vec \eta(t) $$

where $\vec \eta(t)$ is the output of Neural Network

### (3) Neural Network 

We Let there are N inputs and equal number of Neurons in Hidden Layer as well. This gurrantes minimum error in Least Squares Fitting.

Let $\vec t = [t_1, t_2, ... , t_k]$ denote input vector of Neural Network containing various time points

Considering Calculations for single time point $t_k = t$

Let $w_j$ represents weights of $j^{th}$ Neuron in Hidden Layer

Let $b_j$ denote the bias of $j^{th}$ Neuron in Hidden Layer

We initialize weights and biases of Hidden Layer Neuron as random values.

Let Output Activation Function of Neurons be $\sin(x)$

The Output of $j^{th}$ Neuron in Hidden Layer is given as

$$ u_j(t) = \sin( t  \cdot  w_j  + b_j) $$

$$ u_j = \sin(z_j) $$


Let $\vec W_x$, $\vec W_y$, $\vec W_z$ represents Output Neurons weight

Let $\vec u = [ u_1, u_2, ... u_j ]$

Output of the Neural Network is given as

$$ x_{pred}(t) = \vec u(t) \cdot \vec W_x $$

$$ y_{pred}(t) = \vec u(t) \cdot \vec W_y $$

$$ z_{pred}(t) = \vec u(t) \cdot \vec W_z $$

Thus

$$ \vec \eta(t) = [x_{pred}, \ y_{pred}, \ z_{pred}]^T $$

In Extreme Learning Method (ELM) we only update the Output Layer Weights using Least Square Fitting.

### (4) Parallel Computation of All Time Steps

Let $\mathbf{t}_{k \times 1} = [t_1, t_2, ... , t_k]^T$ denote input vector of Neural Network containing various time points

Let $\mathbf w_{1 \times j} = [w_1, w_2, ... , w_j]$ denote weights of Hidden Layer Neurons

Let $\mathbf b_{1 \times j} = [b_1, b_2, b_3, ... , b_j]$ denote biases of Hidden Layer Neurons 

Using Matrix Multiplication and Addition, we define

Let $ \mathbf{1_{k\times 1}} = [1, 1, 1 ... 1]^T $ be denoted as $ \mathbf{1_{k}} $

$$ \mathbf Z_{k \times j} = \mathbf t \times \mathbf w  + \mathbf{1_{k} }\times \mathbf{b} =
\begin{bmatrix}
t_1w_1 + b_1 & t_1w_2 + b_2 & ... & t_1w_j + b_j\\
t_2w_1 + b_1 & t_2w_2 + b_2 & ... & t_2w_j + b_j \\
t_3w_1 + b_1 & t_3w_2 + b_2 & ... & t_3w_j + b_j \\
... & ... & ... & ... \\
t_kw_1 + b_1 & t_kw_2 + b_2 & ... & t_kw_j + b_j  
\end{bmatrix}
$$

We define $U$ matrix to be elementwise sine of $Z$ matrix

$$ \mathbf U_{k \times j} = \sin(Z_{k \times j}) $$

$$ \mathbf U_{k \times j} =
\begin{bmatrix}
\sin(t_1w_1 + b_1) & \sin(t_1w_2 + b_2) & ... & \sin(t_1w_j + b_j)\\
\sin(t_2w_1 + b_1) & \sin(t_2w_2 + b_2) & ... & \sin(t_2w_j + b_j) \\
\sin(t_3w_1 + b_1) & \sin(t_3w_2 + b_2) & ... & \sin(t_3w_j + b_j) \\
... & ... & ... & ... \\
\sin(t_kw_1 + b_1) & \sin(t_kw_2 + b_2) & ... & \sin(t_kw_j + b_j)  
\end{bmatrix}
$$

Now to Calculate Output of Neural Network, we do Matrix Multiplication with output layer weight vector

Let $\mathbf w^x_{j \times 1} = [w^x_1, w^x_2, ... w^x_j]^T $

Let $\mathbf w^y_{j \times 1} = [w^y_1, w^y_2, ... w^y_j]^T $

Let $\mathbf w^z_{j \times 1} = [w^z_1, w^z_2, ... w^z_j]^T $


$$ \mathbf X_{pred \ ({k \times 1})} = \mathbf U \times \mathbf w^x =  \begin{bmatrix}
\sum_j \sin(t_1 w_j + b_j)w^x_j \\
\sum_j \sin(t_2 w_j + b_j)w^x_j \\
\sum_j \sin(t_3 w_j + b_j)w^x_j \\
...  \\
\sum_j \sin(t_k w_j + b_j)w^x_j  
\end{bmatrix}
$$

In a similar fashion, we define  $ \mathbf Y_{pred \ ({k \times 1})}$ and $ \mathbf Z_{pred \ ({k \times 1})}$


### (5) Linear System of Linear Equations

Our Constraint

$$m \frac{d^2}{dt^2} \vec r(t) = q \vec E + \frac{d\vec r(t)}{dt}  \times \vec B$$

Since $\vec r(t) = \vec r_0 \ + \vec {v_0}t \ + \vec \eta(t)$, differentiating we get

$$ \vec r(t) = \vec r_0 \ + \vec {v_0}t \ + \vec \eta(t) $$

$$ \frac{d\vec r(t)}{dt} = \vec v_0 \ + \frac{d\vec \eta(t)}{dt} $$

$$ \frac{d^2\vec r(t)}{dt^2} = \frac{d^2\vec \eta(t)}{dt^2} $$

Also From Neural Network, we have $\vec \eta(t)=[x_{pred}, y_{pred}, z_{pred}]^T$

Let $\theta : x,y, z $ then
$$ \theta_{pred}(t) = \vec u \cdot \vec W_\theta $$

Differentiating these values wrt $t$, we get

$$ \theta^{'}_{pred}(t) =   \frac{d\vec u(t)}{dt}\cdot \vec W_\theta $$

$$ \theta^{''}_{pred}(t) =  \frac{d^2\vec u(t)}{dt^2}\cdot \vec W_\theta  $$


Substiuating we get

$$ m \frac{d^2}{dt^2} \vec \eta(t) = q \vec E + \left( \vec v_0  + \frac{d\vec \eta(t)}{dt} \right)  \times \vec B $$


$$ m \begin{bmatrix}
\ddot{\vec{u}}\cdot \vec W_x \\
\ddot{\vec{u}}\cdot \vec W_y \\
\ddot{\vec{u}}\cdot \vec W_z  
\end{bmatrix} = q \vec E + \begin{bmatrix}
v_{x_0} + \dot{\vec{u}}\cdot \vec W_x \\
v_{y_0} + \dot{\vec{u}}\cdot \vec W_y \\
v_{z_0} + \dot{\vec{u}}\cdot \vec W_z  
\end{bmatrix}  \times \vec B $$

Solving for x-component only, others follow the symmetry.

$$ m \ \ddot{\vec{u}}\cdot \vec W_x = q E_x + [(v_{y_0} + \dot{\vec{u}}\cdot \vec W_y)B_z - (v_{z_0} + \dot{\vec{u}}\cdot \vec W_z)B_y ]  $$

Rewriting as

$$ m \ \ddot{\vec{u}}\cdot \vec W_x - B_z\dot{\vec{u}}\cdot \vec W_y + B_y\dot{\vec{u}}\cdot \vec W_z = q E_x + v_{y_0}B_z  - v_{z_0}B_y $$

In Matrix Form as

$$ \begin{pmatrix}
m \ddot{\vec{u}}_T & -B_z \dot{\vec{u}}_T & B_y \dot{\vec{u}}_T
\end{pmatrix} 
\begin{pmatrix}
\vec W_x \\ \vec W_y \\ \vec W_z
\end{pmatrix} 
= q E_x + v_{y_0}B_z  - v_{z_0}B_y $$

By Symmetry, We have a Linear System for Unknown $\mathbf W = (\vec W_x, \vec W_y, \vec W_z)^T$

$$ \begin{bmatrix}
m \ddot{\vec{u}}_T & -B_z \dot{\vec{u}}_T & B_y \dot{\vec{u}}_T \\
m \ddot{\vec{u}}_T & -B_x \dot{\vec{u}}_T & B_z \dot{\vec{u}}_T \\
m \ddot{\vec{u}}_T & -B_y \dot{\vec{u}}_T & B_x \dot{\vec{u}}_T
\end{bmatrix} 
\begin{bmatrix}
\vec W_x \\ \vec W_y \\ \vec W_z
\end{bmatrix} 
= \begin{bmatrix}
q E_x + v_{y_0}B_z  - v_{z_0}B_y \\
q E_y + v_{z_0}B_x  - v_{x_0}B_z \\
q E_z + v_{x_0}B_y  - v_{y_0}B_x
\end{bmatrix}  $$

We can expand the vectors into their elements and create a Large Matrix 

$$ \begin{pmatrix}
m \ddot{\vec{u}}_T & -B_z \dot{\vec{u}}_T & B_y \dot{\vec{u}}_T
\end{pmatrix} 
\begin{pmatrix}
\vec W_x \\ \vec W_y \\ \vec W_z
\end{pmatrix} 
$$

$$ \sum_j^N m \ddot{u}_j W_{x,j} + \sum_j^N - B_z \dot{u}_j W_{y,j} + \sum_j^N B_y \dot{u}_j W_{z,j} = \sum_k^{3N} a_k b_k $$

Thus we have System of type $A_{3 \times 3N}X_{3N \times 1}=Y_{3\times1}$

$$ \begin{bmatrix}
m \ddot{u_1} & ... & m \ddot{u_N} & -B_z \dot{u}_1 & ... & -B_z \dot{u}_N & B_y \dot{u}_1 & ... & B_y \dot{u}_N \\
B_z \dot{u}_1 & ... & B_z \dot{u}_N & m \ddot{u_1} & ... & m \ddot{u_N} & -B_x \dot{u}_1 & ... & -B_x \dot{u}_N \\
-B_y \dot{u}_1 & ... & -B_y \dot{u}_N & B_x \dot{u}_1 & ... & B_x \dot{u}_N & m \ddot{u_1} & ... & m \ddot{u_N} \\
\end{bmatrix} 
\begin{bmatrix}
W_{x,1} \\ ... \\ W_{x,N} \\ W_{j,1} \\ ... \\ W_{j,N} \\ W_{z,1} \\ ... \\ W_{z,N}
\end{bmatrix} $$

$$ = \begin{bmatrix}
q E_x + v_{y_0}B_z  - v_{z_0}B_y \\
q E_y + v_{z_0}B_x  - v_{x_0}B_z \\
q E_z + v_{x_0}B_y  - v_{y_0}B_x
\end{bmatrix}  $$


### (6) Parallel Computation of All Time Steps

The Linear System Derived $A_{3 \times 3N}X_{3N \times 1}=Y_{3\times1}$
We have such $M$ identical systems for each time step ($M$ Steps) $\{ t_1, t_2, ... , t_m \}$

We combine all such equations into single Big Matrix System - $A_{3M \times 3N}X_{3N \times 1}=Y_{3M\times1}$

$$ \begin{bmatrix}
m \ddot{u_1}(t_1) & ... & m \ddot{u_N}(t_1) & -B_z \dot{u}_1(t_1) & ... & -B_z \dot{u}_N(t_1) & B_y \dot{u}_1(t_1) & ... & B_y \dot{u}_N(t_1) \\
B_z \dot{u}_1(t_1) & ... & B_z \dot{u}_N(t_1) & m \ddot{u_1}(t_1) & ... & m \ddot{u_N}(t_1) & -B_x \dot{u}_1(t_1) & ... & -B_x \dot{u}_N(t_1) \\
-B_y \dot{u}_1(t_1) & ... & -B_y \dot{u}_N(t_1) & B_x \dot{u}_1(t_1) & ... & B_x \dot{u}_N(t_1) & m \ddot{u_1}(t_1) & ... & m \ddot{u_N}(t_1) \\
... \\
m \ddot{u_1}(t_M) & ... & m \ddot{u_N}(t_M) & -B_z \dot{u}_1(t_M) & ... & -B_z \dot{u}_N(t_M) & B_y \dot{u}_1(t_M) & ... & B_y \dot{u}_N(t_M) \\
B_z \dot{u}_1(t_M) & ... & B_z \dot{u}_N(t_M) & m \ddot{u_1}(t_M) & ... & m \ddot{u_N}(t_M) & -B_x \dot{u}_1(t_M) & ... & -B_x \dot{u}_N(t_M) \\
-B_y \dot{u}_1(t_M) & ... & -B_y \dot{u}_N(t_M) & B_x \dot{u}_1(t_M) & ... & B_x \dot{u}_N(t_M) & m \ddot{u_1}(t_M) & ... & m \ddot{u_N}(t_M) \\
\end{bmatrix} $$
$$ \times \begin{bmatrix}
W_{x,1} \\ ... \\ W_{x,N} \\ W_{j,1} \\ ... \\ W_{j,N} \\ W_{z,1} \\ ... \\ W_{z,N}
\end{bmatrix} 
= \begin{bmatrix}
q E_x(t_1) + v_{y_0}B_z(t_1)  - v_{z_0}B_y(t_1) \\
q E_y(t_1) + v_{z_0}B_x(t_1)  - v_{x_0}B_z(t_1) \\
q E_z(t_1) + v_{x_0}B_y(t_1)  - v_{y_0}B_x(t_1) \\
... \\
q E_x(t_M) + v_{y_0}B_z(t_M)  - v_{z_0}B_y(t_M) \\
q E_y(t_M) + v_{z_0}B_x(t_M)  - v_{x_0}B_z(t_M) \\
q E_z(t_M) + v_{x_0}B_y(t_M)  - v_{y_0}B_x(t_M)
\end{bmatrix}  $$

### (7) Least Square Solution

The Linear System $A_{3M \times 3N}X_{3N \times 1}=Y_{3M\times1}$ may have no solution, unique solution or infinite solution.

We use Moore–Penrose pseudoinverse Method to construct Best Fit Solution for the system

By SVD, The pseudoinverse of $A$ is given as

$$ A = U \Sigma V^T $$

$$ A^+ = V \Sigma^+ U^T $$

The solution is given as

$$ X \approx A^+Y $$