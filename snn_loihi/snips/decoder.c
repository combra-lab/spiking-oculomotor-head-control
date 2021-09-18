#include <stdlib.h>
#include "nxsdk.h"
#include "decoder.h"

// Spike activity list
int output_spike_activity[output_dim] = {0};

int do_decoder(runState *s){
    return 1;
}

void run_decoder(runState *s){
    int time = s->time_step;

    // Record and add output spikes by reading the memory and reset memory
    for(int ii=0; ii<output_dim; ii++){
        if(SPIKE_COUNT[(time)&3][ii+0x20] > 0){
            output_spike_activity[ii] += 1;
        }
        SPIKE_COUNT[(time)&3][ii+0x20] = 0;
    }

    // Write recorded spike list to host computer through channel
    if(time % decode_window_step == 0){
        int output_channel_id = getChannelID("decodeoutput");
        writeChannel(output_channel_id, output_spike_activity, output_dim);

        // Reset output spike activity list
        for(int ii=0; ii<output_dim; ii++){
            output_spike_activity[ii] = 0;
        }
    }
}