



## [Hopfield Nets](https://www.youtube.com/watch?v=IP3W7cI01VY&list=PLiPvV5TNogxKKwvKb1RKwkq2hm7ZvpHz0&index=11)


+ energy based model 
+ hopfield nets
    + binary threhsold units with recurrent connections between them
    + behavior: oscillation, converge, chaotic
    + if connections are symmetric, there is a global energy function
        + each binary configuration of whole network has an energy
+ energy function 
    + global energy is sum of local contribution 
        + connection weight and binary states of two neurons
        + `E = -sum_i s_i b_i - sum_i<j s_i s_j w_ij`
            + `s_i` state of `i`-th neuron 
            + `b_i,` `w_ij` are weights
        + energy gap (local change affects global energy)
            + `deltaE_i = b_i + sum_j s_j w_ij`
+ settling to (local) energy minimum
    + start from random state
    + sequential update (update 1 unit)
        + update each unit to whichever of its two states gives the lowest global energy 
        + needs to be sequential, because otherwise energy has the potential to go up
+ hopfield (1982) idea
    + (content addressable memory) memories could be energy minima of a neural net
        + an item can be accessed by just knowing part of its content
        + robust against hardware damage
        + analogy: reconstruct a dinosaur from a few bones
    + 








