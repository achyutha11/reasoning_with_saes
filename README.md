# Steering Faithfulness [WiP]
Repository containing code for a project investigating faithfulness using a steering-based approach. 

Steering vectors are obtained by finding the mean difference in residual stream activations at different layers on curated "faithful" and "unfaithful" datasets. The vectors are then multiplied by a scaling factor and added to the model residual stream during text generation. Preliminary results suggest improved faithfulness through this approach when tested with mid-layer steering vectors in the "hint" setting introduced in "Reasoning Models Don't Always Say What They Think" (Chen et al., 2025).
