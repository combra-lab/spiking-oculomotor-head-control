#include "nxsdk.h"
#define output_dim 10  // Output neuron layer dimension
#define decode_window_step 100  // Number of steps for each window

/****************************************************************************************
Decoder SNIP in snip_window_regular record and add spike from output neurons
at every step of Loihi operation and send through decoder channel at every control step.
****************************************************************************************/

int do_decoder(runState *s);
void run_decoder(runState *s);
