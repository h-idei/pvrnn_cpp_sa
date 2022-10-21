//  Predictive-coding-inspired Variational RNN
//  learning.cpp
//  Copyright © 2022 Hayato Idei. All rights reserved.
//
#include "network.hpp"
#include <sys/time.h>

//hyper-parameter for learning
#define EPOCH 200000 // iteration of parameter update in training (set in the main function in error regression) 
#define SAVE_EPOCH 5000
#define SEQ_NUM 48 //sequence number of target data in training or error regression in simulation
#define MAX_LENGTH 200 //used for initializing shape of variables 
#define ALPHA 0.001    //learning rate
#define WEIGHT_DECAY 0.0001
#define ADAM1 0.9
#define ADAM2 0.999

//example of learning
int main(void){ 
    //set path to output file
    string fo_path_param_executive = "./learning_model/executive";
    string fo_path_param_associative = "./learning_model/associative";
    string fo_path_param_proprioceptive = "./learning_model/proprioceptive";
    string fo_path_param_exteroceptive = "./learning_model/exteroceptive";
    string fo_path_param_out_proprio = "./learning_model/out_proprio";
    string fo_path_param_out_extero = "./learning_model/out_extero";
    string fo_path_generation_executive = "./learning_generation/executive";
    string fo_path_generation_associative = "./learning_generation/associative";
    string fo_path_generation_proprioceptive = "./learning_generation/proprioceptive";
    string fo_path_generation_exteroceptive = "./learning_generation/exteroceptive";
    string fo_path_generation_out_proprio = "./learning_generation/out_proprio";
    string fo_path_generation_out_extero = "./learning_generation/out_extero";
    //set target data
    vector<int> length(SEQ_NUM, 0);
    vector<vector<vector<double> > > target(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_num, 0)));
    set_target_data(length, target, "learning_data");// read "target_n.txt"
    vector<vector<vector<double> > > target_proprio(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > target_extero(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(x_extero_num, 0)));
    for(int s=0;s<SEQ_NUM;++s){
        for(int t=0;t<MAX_LENGTH;++t){
            for(int i=0;i<x_num;++i){
                if(i<x_proprio_num){
                    target_proprio[s][t][i] = target[s][t][i];
                }else{
                    target_extero[s][t][i-x_proprio_num] = target[s][t][i];
                }
            }
        }
    }

    //set model
    PVRNNTopLayerLatent executive(SEQ_NUM, MAX_LENGTH, executive_z_num, executive_W, fo_path_param_executive, fo_path_generation_executive, "learning");
    PVRNNLayer associative(SEQ_NUM, MAX_LENGTH, associative_d_num, executive_z_num, associative_z_num, associative_W, associative_tau, fo_path_param_associative, fo_path_generation_associative, "learning");
    //PVRNNTopLayer associative(SEQ_NUM, MAX_LENGTH, associative_d_num, associative_z_num, associative_W, associative_tau, fo_path_param_associative, fo_path_generation_associative, "learning");
    PVRNNLayer proprioceptive(SEQ_NUM, MAX_LENGTH, proprioceptive_d_num, associative_d_num, proprioceptive_z_num, proprioceptive_W, proprioceptive_tau, fo_path_param_proprioceptive, fo_path_generation_proprioceptive, "learning");
    PVRNNLayer exteroceptive(SEQ_NUM, MAX_LENGTH, exteroceptive_d_num, associative_d_num, exteroceptive_z_num, exteroceptive_W, exteroceptive_tau, fo_path_param_exteroceptive, fo_path_generation_exteroceptive, "learning");
    Output out_proprio(SEQ_NUM, MAX_LENGTH, x_proprio_num, proprioceptive_d_num, fo_path_param_out_proprio, fo_path_generation_out_proprio, "learning");
    Output out_extero(SEQ_NUM, MAX_LENGTH, x_extero_num, exteroceptive_d_num, fo_path_param_out_extero, fo_path_generation_out_extero, "learning");
    
    //concatenated variables used in backprop
    vector<double> prop_exte_tau(proprioceptive_d_num+exteroceptive_d_num, 0);
    vector<vector<double> > prop_exte_weight_ld(proprioceptive_d_num+exteroceptive_d_num, vector<double>(associative_d_num, 0)); 
    vector<vector<vector<double> > > prop_exte_grad_internal_state_d(SEQ_NUM, vector<vector<double> >(MAX_LENGTH, vector<double>(proprioceptive_d_num+exteroceptive_d_num, 0)));
    
    //measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    //PVRNN learning
    //reset loss (free-energy)
    double loss=0;
    for(int epoch=0;epoch<EPOCH;++epoch){
        //ramdom sampling
        executive.set_eps();
        associative.set_eps();
        proprioceptive.set_eps();
        exteroceptive.set_eps();
        //multiple processing
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int s=0;s<SEQ_NUM;++s){
            int time_length = length[s];
            //Feedforward
            for(int t=0;t<time_length;++t){
                 executive.forward(t, s, "posterior");
                 associative.forward(executive.z, t, s, epoch, "posterior");
                 //associative.forward(t, s, epoch, "posterior");
                 proprioceptive.forward(associative.d, t, s, epoch, "posterior");
                 exteroceptive.forward(associative.d, t, s, epoch, "posterior");
                 out_proprio.forward(proprioceptive.d, t, s);
                 out_extero.forward(exteroceptive.d, t, s);
            }
            //Backward
            for(int t=time_length-1;t>=0;--t){
                out_extero.backward(exteroceptive.d, target_extero, time_length, t, s);
                out_proprio.backward(proprioceptive.d, target_proprio, time_length, t, s);
                exteroceptive.backward(associative.d, out_extero.weight_ohd, out_extero.grad_internal_state_output, out_extero.tau, time_length, t, s);
                proprioceptive.backward(associative.d, out_proprio.weight_ohd, out_proprio.grad_internal_state_output, out_proprio.tau, time_length, t, s);
                //concatenate some variables in proprioceptive and exteroceptive areas to make backprop-inputs to the associative area
                for(int i=0;i<proprioceptive_d_num+exteroceptive_d_num;++i){
                    if(i<proprioceptive_d_num) {
                        prop_exte_grad_internal_state_d[s][t][i] = proprioceptive.grad_internal_state_d[s][t][i];
                        prop_exte_tau[i] = proprioceptive.tau[i];
                    }else{
                        prop_exte_grad_internal_state_d[s][t][i] = exteroceptive.grad_internal_state_d[s][t][i-proprioceptive_d_num];
                        prop_exte_tau[i] = exteroceptive.tau[i-proprioceptive_d_num];
                    }
                    for(int j=0;j<associative_d_num;++j){
                        if(i<proprioceptive_d_num) {
                            prop_exte_weight_ld[i][j] = proprioceptive.weight_dhd[i][j];
                        }else{
                            prop_exte_weight_ld[i][j] = exteroceptive.weight_dhd[i-proprioceptive_d_num][j];
                        }
                    }
                }
                associative.backward(executive.z, prop_exte_weight_ld, prop_exte_grad_internal_state_d, prop_exte_tau, time_length, t, s);
                //associative.backward(prop_exte_weight_ld, prop_exte_grad_internal_state_d, prop_exte_tau, time_length, t, s);
                executive.backward(associative.weight_dhd, associative.grad_internal_state_d, associative.tau, time_length, t, s);
            }
            //Save generated sequence
            if((epoch) % SAVE_EPOCH == 0){
                executive.save_sequence(time_length, s, epoch);
                associative.save_sequence(time_length, s, epoch);
                proprioceptive.save_sequence(time_length, s, epoch);
                exteroceptive.save_sequence(time_length, s, epoch);
                out_proprio.save_sequence(target_proprio, time_length, s, epoch);
                out_extero.save_sequence(target_extero, time_length, s, epoch);
            }
        }
        loss = 0;
        //sum gradients over sequences
        for(int s=0;s<SEQ_NUM;++s){
            out_extero.sum_gradient(s);
            out_proprio.sum_gradient(s);
            exteroceptive.sum_gradient(s);
            proprioceptive.sum_gradient(s);
            associative.sum_gradient(s);
            //compute free-energy accumulated over time steps and sequences
            int time_length = length[s];
            double normalize = 1.0/SEQ_NUM;
            for(int t=0;t<time_length; ++t){
                for(int i=0; i<x_proprio_num; ++i){
                    loss += out_proprio.prediction_error[s][t][i] * normalize;
                }
                for(int i=0; i<x_extero_num; ++i){
                    loss += out_extero.prediction_error[s][t][i] * normalize;
                }
                for(int i=0; i<exteroceptive_z_num; ++i){
                    loss += exteroceptive.wkld[s][t][i] * normalize;
                }
                for(int i=0; i<proprioceptive_z_num; ++i){
                    loss += proprioceptive.wkld[s][t][i] * normalize;
                }
                for(int i=0; i<associative_z_num; ++i){
                    loss += associative.wkld[s][t][i] * normalize;
                }
                for(int i=0; i<executive_z_num; ++i){
                    loss += executive.wkld[s][t][i] * normalize;
                }
            }
        }
        //print error
        printf("%d: %f\n", epoch, loss);
        
        //Save weight
        if((epoch) % SAVE_EPOCH == 0){
            out_extero.save_parameter(epoch);
            out_proprio.save_parameter(epoch);
            exteroceptive.save_parameter(epoch);
            proprioceptive.save_parameter(epoch);
            associative.save_parameter(epoch);
        }
        //update parameter
        //set constant for Rectified Adam
        double rho_inf = 2.0/(1.0-ADAM2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(ADAM2,epoch+1)/(1.0-pow(ADAM2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/ (rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_radam1 = 1.0/(1.0-pow(ADAM1,epoch+1));
        double l_radam2 = 1.0-pow(ADAM2,epoch+1);
        out_extero.update_parameter_radam(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        out_proprio.update_parameter_radam(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        exteroceptive.update_parameter_radam(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        proprioceptive.update_parameter_radam(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        associative.update_parameter_radam(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2,WEIGHT_DECAY);
        executive.update_parameter_radam(ALPHA,ADAM1,ADAM2,rho,r_store,m_radam1,l_radam2);
        //reset gradient
        //*there is no need to reset gradient in the executive area because the executive area has adaptive vector only (there is no summation of gradients over time steps or sequences).
        associative.reset_gradient();
        proprioceptive.reset_gradient();
        exteroceptive.reset_gradient();
        out_proprio.reset_gradient();
        out_extero.reset_gradient();
    }
    
    //Save trained result
    for(int s=0;s<SEQ_NUM;++s){
            int time_length = length[s];
            //Feedforward
            for(int t=0;t<time_length;++t){
                 executive.forward(t, s, "posterior");
                 associative.forward(executive.z, t, s, EPOCH, "posterior");
                 //associative.forward(t, s, EPOCH, "posterior");
                 proprioceptive.forward(associative.d, t, s, EPOCH, "posterior");
                 exteroceptive.forward(associative.d, t, s, EPOCH, "posterior");
                 out_proprio.forward(proprioceptive.d, t, s);
                 out_extero.forward(exteroceptive.d, t, s);
            }
            //Backward (compute prediction error and KLD)
            for(int t=time_length-1;t>=0;--t){
                out_extero.backward(exteroceptive.d, target_extero, time_length, t, s);
                out_proprio.backward(proprioceptive.d, target_proprio, time_length, t, s);
                exteroceptive.backward(associative.d, out_extero.weight_ohd, out_extero.grad_internal_state_output, out_extero.tau, time_length, t, s);
                proprioceptive.backward(associative.d, out_proprio.weight_ohd, out_proprio.grad_internal_state_output, out_proprio.tau, time_length, t, s);
                //concatenate some variables in proprioceptive and exteroceptive areas to make backprop-inputs to the associative area
                for(int i=0;i<proprioceptive_d_num+exteroceptive_d_num;++i){
                    if(i<proprioceptive_d_num) {
                        prop_exte_grad_internal_state_d[s][t][i] = proprioceptive.grad_internal_state_d[s][t][i];
                        prop_exte_tau[i] = proprioceptive.tau[i];
                    }else{
                        prop_exte_grad_internal_state_d[s][t][i] = exteroceptive.grad_internal_state_d[s][t][i-proprioceptive_d_num];
                        prop_exte_tau[i] = exteroceptive.tau[i-proprioceptive_d_num];
                    }
                    for(int j=0;j<associative_d_num;++j){
                        if(i<proprioceptive_d_num) {
                            prop_exte_weight_ld[i][j] = proprioceptive.weight_dhd[i][j];
                        }else{
                            prop_exte_weight_ld[i][j] = exteroceptive.weight_dhd[i-proprioceptive_d_num][j];
                        }
                    }
                }
                associative.backward(executive.z, prop_exte_weight_ld, prop_exte_grad_internal_state_d, prop_exte_tau, time_length, t, s);
                //associative.backward(prop_exte_weight_ld, prop_exte_grad_internal_state_d, prop_exte_tau, time_length, t, s);
                executive.backward(associative.weight_dhd, associative.grad_internal_state_d, associative.tau, time_length, t, s);
            }
            //Save generated sequence
            executive.save_sequence(time_length, s, EPOCH);
            associative.save_sequence(time_length, s, EPOCH);
            proprioceptive.save_sequence(time_length, s, EPOCH);
            exteroceptive.save_sequence(time_length, s, EPOCH);
            out_proprio.save_sequence(target_proprio, time_length, s, EPOCH);
            out_extero.save_sequence(target_extero, time_length, s, EPOCH);
    }
    out_extero.save_parameter(EPOCH);
    out_proprio.save_parameter(EPOCH);
    exteroceptive.save_parameter(EPOCH);
    proprioceptive.save_parameter(EPOCH);
    associative.save_parameter(EPOCH);
    
    gettimeofday(&end, NULL);
    float delta = end.tv_sec  - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("time %lf[s]\n", delta);
    
    return 0;
}
