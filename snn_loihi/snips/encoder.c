#include <stdlib.h>
#include "nxsdk.h"
#include "encoder.h"

// Input current list
int input_current[input_dim] = {0};

// Input voltage list
int input_voltage[input_dim] = {0};

int do_encoder(runState *s){
    return 1;
}

void run_encoder(runState *s){
    int time = s->time_step;

    // Read input current for each window at the start of the window (time start at 1)
    if(time % encode_window_step == 1){
        int input_channel_id = getChannelID("encodeinput");
        readChannel(input_channel_id, &input_current, input_dim);
    }

    // Generate spikes base on input current (Input neurons as Integrate-and-Fire neurons with soft-reset)
    for(int ii=0; ii<input_dim; ii++){
        input_voltage[ii] += input_current[ii];
        if(input_voltage[ii] > input_voltage_threshold){
            input_voltage[ii] -= input_voltage_threshold;
            int input_axon_id = input_axon_id_start + ii;
            uint16_t axonId = 1<<14 | ((input_axon_id) & 0x3FFF);
            ChipId chipId = nx_nth_chipid(input_chip);
            nx_send_remote_event(time, chipId, (CoreId){.id=4+input_core}, axonId);
        }
    }
}
