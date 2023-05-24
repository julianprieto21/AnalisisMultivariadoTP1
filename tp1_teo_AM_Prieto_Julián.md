# **Trabajo Práctico Nro. 1**
## **Analisis Multivariado**
### Prieto Julián &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;45065709
<br>
<br>

## Por terminar:
- Revisar hessiano y la comprobacion de si es semi definido positivo (punto 1.a)
- Estimar sigma (Punto 1.d)
- 
<br>
<br>

# **1.a** 

Calcular gradiente y hessiano de:

$$ 
J(\theta)=-\frac{1}{m} \sum_{i*1}^{m} \left[ y^{(i)} \log(h_\theta(x_i)) + (1 - y^{(i)}) \log(1 - h_\theta(x_i)) \right] 
$$

$\text{Teniendo en cuenta que} h_\theta(x_i)=g(\theta^Tx_i) $,

### **Gradiente:**
$$
\nabla J(\theta)=-\frac{1}{m} \sum_{i*1}^{m} \frac{y^{(i)}}{g(\theta^Tx_i)} g'(\theta^Tx_i) - \frac{1-y^{(i)}}{1-g(\theta^Tx_i)} g'(\theta^Tx_i)$$
$$
\nabla J(\theta)=-\frac{1}{m} \sum_{i*1}^{m} \left[ \frac{y^{(i)}}{g(\theta^Tx_i)} - \frac{1-y^{(i)}}{1-g(\theta^Tx_i)} \right] g'(\theta^Tx_i)
$$
Ya que $ \frac{\partial}{\partial\theta} g(\theta^Tx_i)=g(\theta^Tx_i) (1-g(\theta^Tx_i)) x_i$,
$$
\nabla J(\theta)=-\frac{1}{m} \sum_{i*1}^{m} \left[ \frac{y^{(i)}}{g(\theta^Tx_i)} - \frac{1-y^{(i)}}{1-g(\theta^Tx_i)} \right] g(\theta^Tx_i) (1-g(\theta^Tx_i)) x_i
$$
$$
\nabla J(\theta)=-\frac{1}{m} \sum_{i*1}^{m} \left[ \frac{y^{(i)} g(\theta^Tx_i) (1-g(\theta^Tx_i))}{g(\theta^Tx_i)} - \frac{(1-y^{(i)}) g(\theta^Tx_i) (1-g(\theta^Tx_i))}{1-g(\theta^Tx_i)} \right] x_i
$$
$$
\nabla J(\theta)=-\frac{1}{m} \sum_{i*1}^{m} \left[ y^{(i)}(1 - g(\theta^Tx_i)) - (1-y^{(i)})g(\theta^Tx_i) \right] x_i
$$
$$
\nabla J(\theta)=-\frac{1}{m}\sum_{i*1}^{m}\left[y^{(i)} - y^{(i)}g(\theta^Tx_i) - g(\theta^Tx_i) + y^{(i)} g(\theta^Tx_i)\right]x_i^{(i)}
$$
$$
\nabla J(\theta)=-\frac{1}{m}\sum_{i*1}^{m}\left[y^{(i)} - g(\theta^Tx_i)\right]x_i
$$
$$
\boxed{\nabla J(\theta)=\frac{1}{m}\sum_{i*1}^{m}\left[g(\theta^Tx_i) - y^{(i)} \right]x_i}
$$

### **Hessiano:**
$$
\nabla^2J(\theta)=\frac{1}{m}\sum_{i*1}^{m}\left[ g(\theta^Tx_i) \right]'x_i
$$
$$
\boxed{\nabla^2J(\theta)=\frac{1}{m}\sum_{i*1}^{m}g(\theta^Tx_i) (1-g(\theta^Tx_i)) x_i^T x_i}
$$

### Es semi-definido positivo?
## TERMINAR
<br>
<br>

# **1.c**
Mostrar que
$$
p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1 + \exp\left(-(\theta^T x + \theta_0)\right)}
$$
$\text{donde } \theta \in \mathbb{R}^n \text{ y }\theta_0 \in \mathbb{R} \text{ son funciones de } \phi, \mu_0, \mu_1 \text{ y } \Sigma. $
<br>
<br>

$$
p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{p(x|y=1)\cdot p(y=1)}{p(x)}
$$
$$
p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{p(x|y=1)\cdot p(y=1)}{p(x|y=1)\cdot p(y=1)\cdot p(x|y=0)\cdot p(y=0)}
$$
$\text{diviendo numerador y denominador por }p(x|y=1)\cdot p(y=1)$,
$$
p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1+ \frac{p(x|y=0)\cdot p(y=0)}{p(x|y=1)\cdot p(y=1)}}
$$
$\text{reemplazando por funcion de probabilidad,}$
$$
p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1+ \frac{\frac{1}{{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}} \exp \left( -\frac{1}{2} (x - \mu_0)^T\Sigma^{-1}(x - \mu_0) \right)\cdot (1-\phi)}{\frac{1}{{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}} \exp \left( -\frac{1}{2} (x - \mu_1)^T\Sigma^{-1}(x - \mu_1) \right)\cdot \phi}}
$$
$\text{por propiedad }\frac{a^n}{a^m}=a^{n-m}$,$\text{ y cancelando termino constante,}$
$$
= \frac{1}{1+ \exp \left( -\frac{1}{2} (x - \mu_0)^T\Sigma^{-1}(x - \mu_0) +\frac{1}{2} (x - \mu_1)^T\Sigma^{-1}(x - \mu_1) \right)\frac{(1-\phi)}{\phi}}
$$
$$
= \frac{1}{1+ \exp \left( -\frac{1}{2} (x - \mu_0)^T\Sigma^{-1}(x - \mu_0) +\frac{1}{2} (x - \mu_1)^T\Sigma^{-1}(x - \mu_1) + \ln{\frac{(1-\phi)}{\phi}}\right)}
$$
$\text{aplicando distributiva}$,
$$
= \frac{1}{1+ \exp \left( -\frac{1}{2} (x^T\Sigma^{-1}-\mu_0^T\Sigma^{-1})(x - \mu_0) +\frac{1}{2} (x^T\Sigma^{-1}-\mu_1^T\Sigma^{-1})(x - \mu_1) + \ln{\frac{(1-\phi)}{\phi}}\right)}
$$
$$
= \frac{1}{1+ \exp \left( -\frac{1}{2} (x^T\Sigma^{-1}x-x^T\Sigma^{-1}\mu_0-\mu_0^T\Sigma^{-1}x+\mu_0^T\Sigma^{-1}\mu_0) +\frac{1}{2} (x^T\Sigma^{-1}x-x^T\Sigma^{-1}\mu_1-\mu_1^T\Sigma^{-1}x+\mu_1^T\Sigma^{-1}\mu_1) + \ln{\frac{(1-\phi)}{\phi}}\right)}
$$
$ \text{ya que }x^T\Sigma^{-1}\mu_1=\mu_1^T\Sigma^{-1}x$
$$
= \frac{1}{1+ \exp \left( -\frac{1}{2} (x^T\Sigma^{-1}x-2x^T\Sigma^{-1}\mu_0+\mu_0^T\Sigma^{-1}\mu_0) +\frac{1}{2} (x^T\Sigma^{-1}x-2x^T\Sigma^{-1}\mu_1+\mu_1^T\Sigma^{-1}\mu_1) + \ln{\frac{(1-\phi)}{\phi}}\right)}
$$
$$
= \frac{1}{1+ \exp \left( \frac{-x^T\Sigma^{-1}x}{2}+x^T\Sigma^{-1}\mu_0-\frac{\mu_0^T\Sigma^{-1}\mu_0}{2} +\frac{-x^T\Sigma^{-1}x}{2}+x^T\Sigma^{-1}\mu_1-\frac{\mu_1^T\Sigma^{-1}\mu_1}{2} + \ln{\frac{(1-\phi)}{\phi}}\right)}
$$
$$
= \frac{1}{1+ \exp \left( x^T\Sigma^{-1}\mu_0-\frac{\mu_0^T\Sigma^{-1}\mu_0}{2} +x^T\Sigma^{-1}\mu_1-\frac{\mu_1^T\Sigma^{-1}\mu_1}{2} + \ln{\frac{(1-\phi)}{\phi}}\right)}
$$
$$
\boxed{= \frac{1}{1+ \exp \left( x^T\Sigma^{-1}(\mu_0-\mu_1)-\frac{1}{2}\Sigma^{-1}(\mu_0-\mu_1)+\ln{\frac{(1-\phi)}{\phi}} \right)}}
$$
$\text{con }\theta=\Sigma^{-1}(\mu_1-\mu_0) \text{ y con }\theta_0=\frac{1}{2}\Sigma^{-1}(\mu_0-\mu_1)-\ln{\frac{(1-\phi)}{\phi}}$

$\text{se llega a la expresion:}$
$$
p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1 + \exp\left(-(\theta^T x + \theta_0)\right)}
$$
<br>
<br>

# **1.d**

$$
l(\phi, \mu_0, \mu_1, \Sigma) = \prod_{i=1}^m \ln p\left(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma\right)
$$
$$
= \prod_{i=1}^m \ln p\left(x^{(i)} \,|\, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma\right) p(y^{(i)}; \phi)
$$
$$
= \sum_{i=1}^m \ln p\left(x^{(i)} \,|\, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma\right) + \ln{p(y^{(i)}; \phi)}
$$
$$
= \sum_{i=1}^m \ln \left[\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp(-\frac{1}{2}(x-\mu_i)^T\Sigma^{-1}(x-\mu_i))\right] + \ln\left[\phi^{y^{(i)}}(1-\phi)^{1-y^{(i)}}\right]
$$
$$
= \sum_{i=1}^m \ln \left[\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\right]-\frac{1}{2}(x-\mu_i)^T\Sigma^{-1}(x-\mu_i) + y^{(i)}\ln{(\phi)} + (1-y^{(i)})\ln{(1-\phi)}
$$
$$
= \sum_{i=1}^m \ln{(1)}-\ln{\left[(2\pi)^{n/2}|\Sigma|^{1/2}\right]}-\frac{1}{2}(x-\mu_i)^T\Sigma^{-1}(x-\mu_i) + y^{(i)}\ln{(\phi)} + (1-y^{(i)})\ln{(1-\phi)}
$$
$$
= \sum_{i=1}^m -\ln{\left[(2\pi)^{n/2}\right]}-\ln{\left[{|\Sigma|^{1/2}}\right]}-\frac{1}{2}(x-\mu_i)^T\Sigma^{-1}(x-\mu_i) + y^{(i)}\ln{(\phi)} + (1-y^{(i)})\ln{(1-\phi)}
$$
$$
\boxed{= \sum_{i=1}^m -\frac{n}{2}\ln{(2\pi)}-\frac{1}{2}\ln{(|\Sigma|)}-\frac{1}{2}(x-\mu_i)^T\Sigma^{-1}(x-\mu_i) + y^{(i)}\ln{(\phi)} + (1-y^{(i)})\ln{(1-\phi)}}
$$

$\text{Ahora debemos derivar parcialmente e igualar a 0 para cada parametro}$

$$
\frac{\partial l}{\partial\phi}=\left[y^{(i)}\ln{(\phi)} + (1-y^{(i)})\ln{(1-\phi)}\right]'
$$
$$
\frac{\partial l}{\partial\phi}=\sum_{i=1}^{m} \left[ \frac{y^{(i)}}{\phi} - \frac{1-y^{(i)}}{1-\phi} \right]
$$
$$
\frac{\partial l}{\partial\phi}=\sum_{i=1}^{m} \left[ \frac{y^{(i)} \phi (1-\phi)}{\phi} - \frac{(1-y^{(i)}) \phi (1-\phi)}{1-\phi} \right]
$$
$$
\frac{\partial l}{\partial\phi}=\sum_{i=1}^{m} \left[ y^{(i)} (1-\phi) - (1-y^{(i)}) \phi \right]
$$
$$
\frac{\partial l}{\partial\phi}=\sum_{i=1}^{m} \left[ y^{(i)} -y^{(i)}\phi-\phi+y^{(i)}\phi\right]
$$
$$
\frac{\partial l}{\partial\phi}=\sum_{i=1}^{m} \left[ y^{(i)}-\phi\right]
=
\sum_{i=1}^{m} y^{(i)}-\sum_{i=1}^{m}\phi\
$$
$$
\frac{\partial l}{\partial\phi}=\sum_{i=1}^{m} y^{(i)}-\sum_{i*1}^{m}\phi\
=
\boxed{
\sum_{i=1}^{m} y^{(i)}-m\cdot\phi}
$$
$$
\sum_{i=1}^{m} y^{(i)}-m\cdot\phi = 0
$$
$$
-m\cdot\phi = -\sum_{i=1}^{m}y^{(i)}
$$
$$
\phi = \frac{1}{m}\sum_{i=1}^{m}y^{(i)}
=
\boxed{\frac{1}{m}\sum_{i=1}^{m}1\{y=i\}}
$$


Con respecto a $\mu_i$:
$$
\frac{\partial l}{\partial\mu_i}=\left[-\frac{1}{2}\sum_{i=1}^{m}(x_i-\mu_{y^{i}})^T\Sigma^{-1}(x_i-\mu_{y^{i}})\right]'
$$
$$
\frac{\partial l}{\partial\mu_i}=\left[-\frac{1}{2}\sum_{i=1}^{m}(x_i-\mu_{y^{i}})^T\Sigma^{-1}(x_i-\mu_{y^{i}})\right]'
$$
aplicando distributiva, y ya que $\mu_{y^{i}}^T\cdot\mu_{y^{i}}=\mu_{y^{i}}^2$
$$
\frac{\partial l}{\partial\mu_{y^{i}}}=\left[-\frac{1}{2}\sum_{i=1}^{m}x_i^T\Sigma^{-1}x_i-2\mu_{y^{i}}^T\Sigma^{-1}x_i+\Sigma^{-1}\mu_{y^{i}}^2\right]'
$$
$$
\frac{\partial l}{\partial\mu_{y^{i}}}=-\frac{1}{2}\sum_{i=1}^{m}-2\Sigma^{-1}x_i+2\Sigma^{-1}\mu_{y^{i}}
$$
aplicando nuevamente distributiva,
$$
\boxed{\frac{\partial l}{\partial\mu_{y^{i}}}=\sum_{i=1}^{m}\Sigma^{-1}x_i-\sum_{i=1}^{m}\Sigma^{-1}\mu_{y^{i}}}
$$
<br>

$$
\sum_{i=1}^{m}\Sigma^{-1}x_i-\sum_{i=1}^{m}\Sigma^{-1}\mu_{y^{i}}=0
$$
$$
\sum_{i=1}^{m}1 \{y=y^{(i)}\}\Sigma^{-1}x_i-\sum_{i=1}^{m}1 \{y=y^{(i)}\}\Sigma^{-1}\mu_{y^{i}}=0
$$
$$
\Sigma^{-1}\sum_{i=1}^{m}1 \{y=y^{(i)}\}x_i-\Sigma^{-1}\sum_{i=1}^{m}1 \{y=y^{(i)}\}\mu_{y^{i}}=0
$$
$$
\Sigma^{-1}\sum_{i=1}^{m}1 \{y=y^{(i)}\}x_i=\Sigma^{-1}\sum_{i=1}^{m}1 \{y=y^{(i)}\}\mu_{y^{i}}
$$
$$
\frac{\Sigma^{-1}\sum_{i=1}^{m}1 \{y=y^{(i)}\}x_i}{\Sigma^{-1}\sum_{i=1}^{m}1 \{y=y^{(i)}\}}=\mu_{y^{i}}
$$
$$
\frac{\sum_{i=1}^{m}1 \{y=y^{(i)}\}x_i}{\sum_{i=1}^{m}1 \{y=y^{(i)}\}}=\mu_{y^{i}}
$$

Osea, que nos quedaria de la siguiente manera:
$$
\boxed{
\mu_0=\frac{\sum_{i=1}^{m}1 \{y=0\}x_i}{\sum_{i=1}^{m}1 \{y=0\}}
\text{ y }
\mu_1=\frac{\sum_{i=1}^{m}1 \{y=1\}x_i}{\sum_{i=1}^{m}1 \{y=1\}}}
$$


Con respecto a $\Sigma$:
$$
\frac{\partial l}{\partial\Sigma}=\sum_{i=1}^{m}\left[-\frac{1}{2}\ln|\Sigma|-\frac{1}{2}(x_i-\mu_{y^{i}})^T\Sigma^{-1}(x_i-\mu_{y^{i}})\right]'
$$
$$
\frac{\partial l}{\partial\Sigma}=-\frac{1}{2}\sum_{i=1}^{m}\left[\ln|\Sigma|+(x_i-\mu_{y^{i}})^T\Sigma^{-1}(x_i-\mu_{y^{i}})\right]'
$$
#### TERMINAR

# **2.a**
