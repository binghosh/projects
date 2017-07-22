#include "mbed.h"
#define rise_edge 0
#define fall_edge 1
 
void timer_meas(bool fall_rise);
 
class AMEnvelope {
public:
    AMEnvelope(PinName pin) : _interrupt(pin) {        // create the InterruptIn on the pin specified to Counter
        _interrupt.rise(this, &AMEnvelope::rising); // attach increment function of this counter instance
     _interrupt.fall(this, &AMEnvelope::falling); // attach increment function of this counter instance

    }
 
    void rising() {
       timer_meas(rise_edge);
    }
     void falling() {
        timer_meas(fall_edge);
    }
 
private:
    InterruptIn _interrupt;
};
 
AMEnvelope AMEnvelope(5);
 Timer AM;
 struct frame {
    bool SOF;
    int timer_val;
    int last_timer;
};
struct frame frame_byte;

 bool timer_running=false;
int main() {
    AM.start();
    frame_byte.SOF = false;
    timer_running=false;
    while(1) {
        if( frame_byte.SOF==true){
            wait(3);
            led1=0;
    }
}

void timer_meas(bool fall_rise) {
    if(fall_rise == fall_edge&&timer_running ==false){
        AM.reset();
        timer_running = true;
     } else if(frame_byte.SOF == false){
            frame_byte.timer_val=AM.read_us();
            switch (fall_rise) {
                case fall_edge:

                    if(frame_byte.last_timer == 10&&((40<= frame_byte.timer_val)&&(frame_byte.timer_val <=70))){
                        led1 = 1;
                        frame_byte.SOF=true;
                        } else{
                           timer_running=false; 
                        }
                  break;
                case rise_edge:
                    if(frame_byte.last_timer == 0&&((5<= frame_byte.timer_val)&&(frame_byte.timer_val <=15))){
                        frame_byte.last_timer = 10;
                        }else{
                        timer_running=false;
                        }
                  break;
                default: 
                  timer_running=false;
                break;
                } 
         }
}
