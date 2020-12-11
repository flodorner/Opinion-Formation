# Modelling Multi-dimensionalOpinion and Polarization Formation 
# (Agent-Based Modeling and Social System Simulation 2020)


> * Group Name: The Opinioneers
> * Group participants names: Michael Andres, Florian Dorner, Gian Luca Gehwolf, Fabian Hafner, David Metzger
> * Project Title: Modelling Multi-dimensinal Opinions and Polarization Formation

## General Introduction

Political polarization in democratic societies has reached a point, where it poses a threat to political stability. The (re-)emergence of poulism in certain countries in Europe and in the US has led to diveded societies with strongly opposing opinions. We are interested to study how opinion polarization is generated. In adddtion, we are interested to see how artificial actors, so-called bots, have an impact on opinion polarization as in the past there have been made claims that bots have tried to influence recent presidential election campaigns. There are many opinion and polarization dynamics model that try to simulate how opinions are formed and how polarization can result form this.
Holme & Newman (2006) developed a opinion dynamics model that lets networks and opinions simultaneously influence each other. On the other hand, Schweighofer et al. (2020) devoloped a multiopinion dynamics model, called weighet balance model (WBT), that can generate polarization, but does not account for the influence of network structures. We are interested to combine this two models to a "generalized WBT model".

## The Model : Generalized WBT model
* Dependent variables:
  * Hyperpolarization (H): The simultaneous emergence of the socio-political phenomena of opinion extremeness and issue constrain.
     * What do we want to study:
        * The effect of a coevolution of network structures and opinion fomration on hyperpolarization
        * The effect of introducing bots to the generalised WBT model
     * How we want to measure it: 
        * Hyperpolarization is measured as suggested in the WBT model developed by Schweighofer et al. (2020). It can be best described as a generalized variance of opinions normalized to 0,1, where max. hyperpolarization is reached when H = 1. 

**Note:** The effect of the indepenent variables on hyperpolarization is evaluated for the generalized WBT model (network structures included) **and** for the case where bots are introduced (generalized WBT model + bots).
  
* Independent variables: 
   *  The attitude transformation f function: The  functional  form  of f is  chosen  to  preserve  the  sign  to  reflect  peoples’  tendency  to agree with people they like and disagree with those they do not like – a phenomenon often referred to as the backfire-effect.
      * What do we want to study:
        * The effect different f functions have on hyperpolarization
      * How we want to measure it: 
        * Leave all other parameters constant and use different f functions and evaluate how hyperpolarization is affected
        
   * Evaluative extremness e: How strongly small amounts of (dis-)agreement are reflected in peoples’ displayed attitude towards each other
     * What do we want to study:
        * The effect different values [0,1] of e have on hyperpolarization
     * How we want to measure it: 
        * Leave all other parameters constant and use different e values and evaluate how hyperpolarization is affected
        
   * The  distance  parameter ε: Indicates  a  node’s  ”sphere  of influence”.
     * What do we want to study:
        * The effect different values [0,1] of ε have on hyperpolarization
     * How we want to measure it: 
        * Leave all other parameters constant and use different e values and evaluate how hyperpolarization is affected

* Why is it a good model? 
  * We use a polarization dynamics model that can generate polarization through its mechanism and combine it with a model the includes the effects that network strucutres have on opinion formation. Thus, it is more realstic than the WBT model developed by Schweighofer et al. (2020). In addition, it makes more sense to investigate the effects of bots on hyperpolarization by using the generalized WBT model isntead of the the WBT model.
   

## Fundamental Questions

* How does the introdcution of a complex network influence WBT model developed by Schweighofer et a. (2020)?
* How do bots influence this model?

## Expected Results

* The inclusion of a changing network structure allows for different outocomes (no polarization, fragmentation, and hyperpolarization), which strongly depend on the settings of the parameters and the respective chosen update and connect functions of the generalised WBT model. This is a clear difference to the WBT model, which is  only able to generate a polarized state or nothing. As opposed to the WBT model, the generalized WBT  fragmentation instead of hyper-polarization and can be seen as an interesting starting point for the examination of effects of various connectivity configurations.
* Bots can certainly add to hyperpolarization, though it might be necessary thatthey make up half the network for this. Depeding on the settings of the bots (wether the bots have extreme opinions or which vertices in the network they are able to influence, etc.), the degree of polarization varies.
* Certain inherent features of the  WBT model such as the evaluative extremness parameter, the attitude transformation function, and the level of noise have a large impact on qualitative model behaviour and often determine  the  extent  of  hyperpolarization.


## References 

- Petter Holme and Mark EJ Newman. Nonequilibrium phase transition in the coevo-lution of networks and opinions.Physical Review E, 74(5):056108, 2006
- Simon Schweighofer, Frank Schweitzer, and David Garcia. A weighted balance modelof opinion hyperpolarization.Journal of Artificial Societies and Social Simulation,23(3):5,    2020.  ISSN 1460-7425.  doi:  10.18564/jasss.4306.  URLhttp://jasss.soc.surrey.ac.uk/23/3/5.html



## Research Methods
  
We intend to use the Agent-Based Models of opinion and polarization dynamics developed by Holme & Newman (2006) and Schweighofer et a. (2020) as base models. We first implement these two models in Python using our own ideas, i.e. how we think that the model could be implemented and not look how the authors implemented them. This followed by proposing a set of extensions that allows us to combine these two models to the generalized WBT model in order to study the effects of network strucutres on polarization dynamics. We also plan to introduce bots to this model to see how this effects hyperpolarization.

