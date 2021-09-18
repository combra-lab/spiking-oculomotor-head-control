#include "nxsdk.h"
#define input_dim 6  // Input neuron layer dimension
#define input_chip 0  // Input neuron chip id
#define input_core 0  // Input neuron core id
#define input_axon_id_start 0  // Start point for input neuron axon id
#define encode_window_step 100  // Number of steps for each window
#define input_voltage_threshold 99  // Input voltage threshold for spike generation

/***************************************************************************
Encoder SNIP in snip_window_regular read input current from encoder channel
at every control loop step. The input current is then used to generate
spike activities for input neuron layer in steps within encode_window_end.
***************************************************************************/

int do_encoder(runState *s);
void run_encoder(runState *s);