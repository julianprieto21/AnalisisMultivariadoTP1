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
Teniendo en cuenta que $h_\theta(x_i)=g(\theta^Tx_i) $,

### **Gradiente:**
$$
\nabla J(\theta)=-\frac{1}{m} \sum_{i*1}^{m} \frac{y^{(i)}}{g(\theta^Tx_i)} g'(\theta^Tx_i) - \frac{1-y^{(i)}}{1-g(\theta^Tx_i)} g'(\theta^Tx_i)
$$
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
= \frac{1}{1+ \exp \left( x^T\Sigma^{-1}(\mu_0-\mu_1)-\frac{1}{2}\Sigma^{-1}(\mu_0-\mu_1)+\ln{\frac{(1-\phi)}{\phi}} \right)}
$$
$\text{con }\theta=\Sigma^{-1}(\mu_1-\mu_0) \text{ y con }\theta_0=\frac{1}{2}\Sigma^{-1}(\mu_0-\mu_1)-\ln{\frac{(1-\phi)}{\phi}}$

$\text{se llega a la expresion:}$
$$
p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1 + \exp\left(-(\theta^T x + \theta_0)\right)}
$$
<br>
<br>

# **1.d**

