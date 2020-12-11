# Agent-Based Modeling and Social System Simulation 2020


> * Group Name: The Opinioneers
> * Group participants names: Michael Andres, Florian Dorner, Gian Luca Gehwolf, Fabian Hafner, David Metzger
> * Project Title: Modelling Multi-dimensinal Opinions and Polarization

## General Introduction

Political polarization in democratic societies has reached a point, where it poses a threat to political stability. The (re-)emergence of poulism in certain countries in Europe and in the US has led to diveded societies with strongly opposing opinions. We are interested to study how opinion polarization is generated. In adddtion, we are interested to see how artificial actors, so-called bots, have an impact on opinion polarization as in the past there have been made claims that bots have tried to influence recent presidential election campaigns. There are many opinion and polarization dynamics model that try to simulate how opinions are formed and how polarization can result form this.
Holme & Newman (2006) developed a opinion dynamics model that lets networks and opinions simultaneously influence each other. On the other hand, Schweighofer et al. (2020) devoloped a multiopinion dynamics model, called weighet balance model (WBT), that can generate polarization, but does not account for the influence of network structures. Therefore, we want to study what effects network structures have on polarization dynamics. In addition, we are interested to see what effects the introduction of bots can have on a multidimensional opinion dynamics model that can generate polarization and acocunts for complex network structures.

(States your motivation clearly: why is it important / interesting to solve this problem?)
(Add real-world examples, if any)
(Put the problem into a historical context, from what does it originate? Are there already some proposed solutions?)

## The Model
What we want to study?
* Coevolution of networks and opinions model:
  * Dependent variable: 
    - Connected component community sizes
  * Independedt variables: 
    - Number of edges --> 
    - Connect: with probability φ, a random edge connected to a vertex n,(n,m) is selected and changed to (n,m′) wherevm′is a vertexvwith the same opinion as n, thus o(m′)     =o(n).  If no such m′exists, nothing happens.
    - Update: with probability 1−φ, a random vertexl that is directly connected to n by an edge is selected and n’s opiniono(n) is set too(l).
  * The model allows that network structures (connect) and opinion change (update) evovle at the same the same time and influence each other.
  
* WBT model:
  * Dependent variable:
    - Hyperpolarization H(O), where O is the  opinion matrice consisting of D * N opinions ( D = number of opinions, N = number of agents)
  * Independent variable: 
    - o(i): opinion of agent i  



(Define dependent and independent variables you want to study. Say how you want to measure them.) (Why is your model a good abtraction of the problem you want to study?) (Are you capturing all the relevant aspects of the problem?)


## Fundamental Questions

* How does the introdcution of a complex network influence WBT model developed by Schweighofer et a. (2020)?
* How do bots influence this model?

(At the end of the project you want to find the answer to these questions)
(Formulate a few, clear questions. Articulate them in sub-questions, from the more general to the more specific. )


## Expected Results

(What are the answers to the above questions that you expect to find before starting your research?)
* The inclusion of a changing network structure allows for different outocomes (no polarization, fragmentation, and hyperpolarization), which strongly depend on the settings of the parameters and the respective chosen update and connect functions of the generalised WBT model. This is a clear difference to the WBT model, which is  only able to generate a polarized state or nothing. As opposed to the WBT model, the generalized WBT  fragmentation instead of hyper-polarization and can be seen as an interesting starting point for the examination of effects of various connectivity configurations.
* Bots can certainly add to hyperpolarization, though it might be necessary thatthey make up half the network for this. Depeding on the settings of the bots (wether the bots have extreme opinions or which vertices in the network they are able to influence, etc.), the degree of polarization varies.
* Certain inherent features of the  WBT model such as the evaluative extremness parameter, the attitude transformation function, and the level of noise have a large impact on qualitative model behaviour and often determine  the  extent  of  hyperpolarization.


## References 

(Add the bibliographic references you intend to use)
(Explain possible extension to the above models)
(Code / Projects Reports of the previous year)

- Petter Holme and Mark EJ Newman. Nonequilibrium phase transition in the coevo-lution of networks and opinions.Physical Review E, 74(5):056108, 2006
- Simon Schweighofer, Frank Schweitzer, and David Garcia. A weighted balance modelof opinion hyperpolarization.Journal of Artificial Societies and Social Simulation,23(3):5,    2020.  ISSN 1460-7425.  doi:  10.18564/jasss.4306.  URLhttp://jasss.soc.surrey.ac.uk/23/3/5.html



## Research Methods

(Cellular Automata, Agent-Based Model, Continuous Modeling...) (If you are not sure here: 1. Consult your colleagues, 2. ask the teachers, 3. remember that you can change it afterwards)
  
We intend to use the Agent-Based Models of opinion and polarization dynamics developed by Holme & Newman (2006) and Schweighofer et a. (2020) as base models. We first implement these two models in Python using our own ideas, i.e. how we think that the model could be implemented and not look how the authors implemented them. This followed by proposing a set of extensions that allows us to combine these two models to a "generalized" model in order to study the effects of network strucutres on polarization dynamics. We also plan to introduce bots to this "generalized" model to see how this changes the opinion and polarization dynamics.


## Other

(mention datasets you are going to use)
