//  Predictive-coding-inspired Variational RNN
//  error_regression.cpp
//  Copyright © 2022 Hayato Idei. All rights reserved.
//
#include "network.hpp"
#include <sys/time.h>

//hyper-parameter used in error regression
#define TEST_SEQ_NUM 1 //sequence number of target data in error regression
#define ADAM1 0.9
#define ADAM2 0.999
#define TEST_DATA_LENGTH 200 //if test data is externally given
#define TEST_SEQ_NUM_LEARNING 48 //sequence number of training data (used for initializing adaptive vectors with those acquired for training data)
#define TEST_MAX_LENGTH_LEARNING 200 //max time length of training data (used for initializing adaptive vectors with those acquired for training data)

//example of error regression in simulation (in which target data is defined in advence)
int main(void){ 
    //set target data
    vector<int> length(TEST_SEQ_NUM, 0);
    vector<vector<vector<double> > > target(TEST_SEQ_NUM, vector<vector<double> >(TEST_DATA_LENGTH, vector<double>(x_num, 0)));
    set_target_data(length, target, "test_data"); //read "target_n.txt"
    vector<vector<vector<double> > > target_proprio(TEST_SEQ_NUM, vector<vector<double> >(TEST_DATA_LENGTH, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > target_extero(TEST_SEQ_NUM, vector<vector<double> >(TEST_DATA_LENGTH, vector<double>(x_extero_num, 0)));
    for(int s=0;s<TEST_SEQ_NUM;++s){
        for(int t=0;t<TEST_DATA_LENGTH;++t){
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
    ER_PVRNNTopLayerLatent executive(TEST_SEQ_NUM, TEST_DATA_LENGTH, executive_z_num, executive_W, "executive");
    ER_PVRNNLayer associative(TEST_SEQ_NUM, TEST_DATA_LENGTH, associative_d_num, executive_z_num, associative_z_num, associative_W, associative_tau, "associative");
    //ER_PVRNNTopLayer associative(TEST_SEQ_NUM, TEST_DATA_LENGTH, associative_d_num, associative_z_num, associative_W, associative_tau, "associative");
    ER_PVRNNLayer proprioceptive(TEST_SEQ_NUM, TEST_DATA_LENGTH, proprioceptive_d_num, associative_d_num, proprioceptive_z_num, proprioceptive_W, proprioceptive_tau, "proprioceptive");
    ER_PVRNNLayer exteroceptive(TEST_SEQ_NUM, TEST_DATA_LENGTH, exteroceptive_d_num, associative_d_num, exteroceptive_z_num, exteroceptive_W, exteroceptive_tau, "exteroceptive");
    ER_Output out_proprio(TEST_SEQ_NUM, TEST_DATA_LENGTH, x_proprio_num, proprioceptive_d_num, "out_proprio");
    ER_Output out_extero(TEST_SEQ_NUM, TEST_DATA_LENGTH, x_extero_num, exteroceptive_d_num, "out_extero");
    
    //concatenated variables used in backprop
    vector<double> prop_exte_tau(proprioceptive_d_num+exteroceptive_d_num, 0);
    vector<vector<double> > prop_exte_weight_ld(proprioceptive_d_num+exteroceptive_d_num, vector<double>(associative_d_num, 0)); 
    vector<vector<vector<double> > > prop_exte_grad_internal_state_d(TEST_SEQ_NUM, vector<vector<double> >(TEST_DATA_LENGTH, vector<double>(proprioceptive_d_num+exteroceptive_d_num, 0)));
    
    vector<int> window_list = {10, 15};
    vector<int> epoch_list = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    vector<double> learning_rate_list = {0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5};
    int window_list_size = int(window_list.size());
    int epoch_list_size = int(epoch_list.size());
    int learning_rate_list_size = int(learning_rate_list.size());
    for(int w=0;w<window_list_size;++w){
        int window = window_list[w];
        for(int itr=0;itr<epoch_list_size;++itr){
            int iteration = epoch_list[itr];
            for(int lr=0;lr<learning_rate_list_size;++lr){
                double alpha = learning_rate_list[lr];
                int sequence_size = int(target.size());
                //make save directory
                stringstream path_generation_executive;
                path_generation_executive << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/executive";
                string str_path_generation_executive = path_generation_executive.str();
                mkdir(str_path_generation_executive.c_str(), 0777);
                stringstream path_generation_associative;
                path_generation_associative << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/associative";
                string str_path_generation_associative = path_generation_associative.str();
                mkdir(str_path_generation_associative.c_str(), 0777);
                stringstream path_generation_proprioceptive;
                path_generation_proprioceptive << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/proprioceptive";
                string str_path_generation_proprioceptive = path_generation_proprioceptive.str();
                mkdir(str_path_generation_proprioceptive.c_str(), 0777);
                stringstream path_generation_exteroceptive;
                path_generation_exteroceptive << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/exteroceptive";
                string str_path_generation_exteroceptive = path_generation_exteroceptive.str();
                mkdir(str_path_generation_exteroceptive.c_str(), 0777);
                stringstream path_generation_out_proprio;
                path_generation_out_proprio << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/out_proprio";
                string str_path_generation_out_proprio = path_generation_out_proprio.str();
                mkdir(str_path_generation_out_proprio.c_str(), 0777);
                stringstream path_generation_out_extero;
                path_generation_out_extero << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/out_extero";
                string str_path_generation_out_extero = path_generation_out_extero.str();
                mkdir(str_path_generation_out_extero.c_str(), 0777);
                
                //set initial adaptive vector in executive area
                vector<vector<vector<double> > > dummy_mu(TEST_SEQ_NUM_LEARNING, vector<vector<double> >(TEST_MAX_LENGTH_LEARNING, vector<double>(executive_z_num, 0)));
                vector<vector<vector<double> > > dummy_sigma(TEST_SEQ_NUM_LEARNING, vector<vector<double> >(TEST_MAX_LENGTH_LEARNING, vector<double>(executive_z_num, 0)));
                for(int s=0;s<TEST_SEQ_NUM_LEARNING;++s){
                    stringstream str_a_mu, str_a_sigma;
                    str_a_mu << "./learning_model/" << "executive" << "/a_mu_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                    str_a_sigma << "./learning_model/" << "executive" << "/a_sigma_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                    set_weight(dummy_mu[s], str_a_mu.str());
                    set_weight(dummy_sigma[s], str_a_sigma.str());
                }
                set_initial_adaptive_vector(executive.a_mu, dummy_mu);
                set_initial_adaptive_vector(executive.a_sigma, dummy_sigma);
                
                //error regression
                for(int s=0;s<TEST_SEQ_NUM;++s){
                    //measure time
                    struct timeval start, end;
                    gettimeofday(&start, NULL);
                    int time_length = length[s];
                    for(int ct=window-1;ct<time_length;++ct){//ct: current time step
                        printf("Window size: %d, Epoch size: %d, Alpha: %lf, Sequence: %d, Time step: %d\n", window, iteration, alpha, s, ct);
                        int initial_regression;
                        (ct==window-1) ? (initial_regression=1) : (initial_regression=0);
                        for(int epoch=0;epoch<iteration;++epoch){
                            //Feedforward
                            for(int wt=ct-window+1;wt<=ct;++wt){//wt: time step within time window
                                int first_window_step, last_window_step;
                                (wt==ct-window+1) ? (first_window_step=1) : (first_window_step=0);
                                (wt==ct) ? (last_window_step=1) : (last_window_step=0);
                                executive.er_forward(wt, s, epoch, first_window_step, "posterior");
                                associative.er_forward(executive.z, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                //associative.er_forward(wt, s, epoch, initial_regression, last_window_step, "posterior");
                                proprioceptive.er_forward(associative.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                exteroceptive.er_forward(associative.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                out_proprio.er_forward(proprioceptive.d, wt, s);
                                out_extero.er_forward(exteroceptive.d, wt, s);
                            }
                            
                            //Backward
                            //set constant for Rectified Adam
                            double rho_inf = 2.0/(1.0-ADAM2)-1.0;
                            double rho = rho_inf-2.0*(epoch+1)*pow(ADAM2,epoch+1)/(1.0-pow(ADAM2,epoch+1));
                            double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/ (rho_inf-4.0)/(rho_inf-2.0)/rho);
                            double m_radam1 = 1.0/(1.0-pow(ADAM1,epoch+1));
                            double l_radam2 = 1.0-pow(ADAM2,epoch+1);
                            for(int wt=ct;wt>=ct-window+1;--wt){
                                int first_window_step, last_window_step;
                                (wt==ct-window+1) ? (first_window_step=1) : (first_window_step=0);
                                (wt==ct) ? (last_window_step=1) : (last_window_step=0);
                                out_extero.er_backward(target_extero, wt, s);
                                out_proprio.er_backward(target_proprio, wt, s);
                                exteroceptive.er_backward(out_extero.weight_ohd, out_extero.grad_internal_state_output, out_extero.tau, last_window_step, wt, s);
                                proprioceptive.er_backward(out_proprio.weight_ohd, out_proprio.grad_internal_state_output, out_proprio.tau, last_window_step, wt, s);
                                //concatenate some variables in proprioceptive and exteroceptive areas to make backprop-inputs to the associative area
                                for(int i=0;i<proprioceptive_d_num+exteroceptive_d_num;++i){
                                    if(i<proprioceptive_d_num){
                                        prop_exte_grad_internal_state_d[s][wt][i] = proprioceptive.grad_internal_state_d[s][wt][i];
                                        prop_exte_tau[i] = proprioceptive.tau[i];
                                    }else{
                                        prop_exte_grad_internal_state_d[s][wt][i] = exteroceptive.grad_internal_state_d[s][wt][i-proprioceptive_d_num];
                                        prop_exte_tau[i] = exteroceptive.tau[i-proprioceptive_d_num];
                                    }
                                    for(int j=0;j<associative_d_num;++j){
                                        if(i<proprioceptive_d_num){
                                            prop_exte_weight_ld[i][j] = proprioceptive.weight_dhd[i][j];
                                        }else{
                                            prop_exte_weight_ld[i][j] = exteroceptive.weight_dhd[i-proprioceptive_d_num][j];
                                        }
                                    } 
                                }
                                associative.er_backward(prop_exte_weight_ld, prop_exte_grad_internal_state_d, prop_exte_tau, last_window_step, wt, s);
                                executive.er_backward(associative.weight_dhd, associative.grad_internal_state_d, associative.tau, last_window_step, wt, s);
                                
                                //update adaptive vector (Rectified Adam)
                                exteroceptive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                proprioceptive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                associative.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                if(wt==ct-window+1){
                                    executive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                }
                            }
                        }
                    }
                    //Save generated sequence
                    executive.er_save_sequence(str_path_generation_executive, time_length, s, 0);
                    associative.er_save_sequence(str_path_generation_associative, time_length, s, 0);
                    proprioceptive.er_save_sequence(str_path_generation_proprioceptive, time_length, s, 0);
                    exteroceptive.er_save_sequence(str_path_generation_exteroceptive, time_length, s, 0);
                    out_proprio.er_save_sequence(str_path_generation_out_proprio, target_proprio, time_length, s, 0);
                    out_extero.er_save_sequence(str_path_generation_out_extero, target_extero, time_length, s, 0);
                    
                    gettimeofday(&end, NULL);
                    float delta = end.tv_sec  - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
                    printf("time %lf[s]\n", delta);
                }
            }
        }
    }
    return 0;
}

/*
//robot error regression
int main(void){ 
    //set target data
    vector<int> length(TEST_SEQ_NUM, 0);
    vector<vector<vector<double> > > target(TEST_SEQ_NUM, vector<vector<double> >(TEST_DATA_LENGTH, vector<double>(x_num, 0)));
    set_target_data(length, target, "test_data"); //read "target_n.txt"
    vector<vector<vector<double> > > target_proprio(TEST_SEQ_NUM, vector<vector<double> >(1, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > joint_angle(TEST_SEQ_NUM, vector<vector<double> >(1, vector<double>(x_proprio_num, 0)));
    vector<vector<vector<double> > > target_extero(TEST_SEQ_NUM, vector<vector<double> >(1, vector<double>(x_extero_num, 0)));
    vector<double> target_extero_dummy(x_extero_num, 0);

    //set model
    ER_PVRNNTopLayerLatent executive(TEST_SEQ_NUM, 1, executive_z_num, executive_W, "executive");
    ER_PVRNNLayer associative(TEST_SEQ_NUM, 1, associative_d_num, executive_z_num, associative_z_num, associative_W, associative_tau, "associative");
    //ER_PVRNNTopLayer associative(TEST_SEQ_NUM, 1, associative_d_num, associative_z_num, associative_W, associative_tau, "associative");
    ER_PVRNNLayer proprioceptive(TEST_SEQ_NUM, 1, proprioceptive_d_num, associative_d_num, proprioceptive_z_num, proprioceptive_W, proprioceptive_tau, "proprioceptive");
    ER_PVRNNLayer exteroceptive(TEST_SEQ_NUM, 1, exteroceptive_d_num, associative_d_num, exteroceptive_z_num, exteroceptive_W, exteroceptive_tau, "exteroceptive");
    ER_Output out_proprio(TEST_SEQ_NUM, 1, x_proprio_num, proprioceptive_d_num, "out_proprio");
    ER_Output out_extero(TEST_SEQ_NUM, 1, x_extero_num, exteroceptive_d_num, "out_extero");
    
    //concatenated variables used in backprop
    vector<double> prop_exte_tau(proprioceptive_d_num+exteroceptive_d_num, 0);
    vector<vector<double> > prop_exte_weight_ld(proprioceptive_d_num+exteroceptive_d_num, vector<double>(associative_d_num, 0)); 
    vector<vector<vector<double> > > prop_exte_grad_internal_state_d(TEST_SEQ_NUM, vector<vector<double> >(1, vector<double>(proprioceptive_d_num+exteroceptive_d_num, 0)));
    
    vector<int> window_list = {10, 15};
    vector<int> epoch_list = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    vector<double> learning_rate_list = {0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5};
    int window_list_size = int(window_list.size());
    int epoch_list_size = int(epoch_list.size());
    int learning_rate_list_size = int(learning_rate_list.size());
    for(int w=0;w<window_list_size;++w){
        int window = window_list[w];
        for(int itr=0;itr<epoch_list_size;++itr){
            int iteration = epoch_list[itr];
            for(int lr=0;lr<learning_rate_list_size;++lr){
                double alpha = learning_rate_list[lr];
                int sequence_size = int(target.size());
                //make save directory
                stringstream path_generation_executive;
                path_generation_executive << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/executive";
                string str_path_generation_executive = path_generation_executive.str();
                mkdir(str_path_generation_executive.c_str(), 0777);
                stringstream path_generation_associative;
                path_generation_associative << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/associative";
                string str_path_generation_associative = path_generation_associative.str();
                mkdir(str_path_generation_associative.c_str(), 0777);
                stringstream path_generation_proprioceptive;
                path_generation_proprioceptive << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/proprioceptive";
                string str_path_generation_proprioceptive = path_generation_proprioceptive.str();
                mkdir(str_path_generation_proprioceptive.c_str(), 0777);
                stringstream path_generation_exteroceptive;
                path_generation_exteroceptive << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/exteroceptive";
                string str_path_generation_exteroceptive = path_generation_exteroceptive.str();
                mkdir(str_path_generation_exteroceptive.c_str(), 0777);
                stringstream path_generation_out_proprio;
                path_generation_out_proprio << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/out_proprio";
                string str_path_generation_out_proprio = path_generation_out_proprio.str();
                mkdir(str_path_generation_out_proprio.c_str(), 0777);
                stringstream path_generation_out_extero;
                path_generation_out_extero << "./test_generation" << "/window_" << window << "/epoch_" << iteration << "/lr_" << alpha << "/out_extero";
                string str_path_generation_out_extero = path_generation_out_extero.str();
                mkdir(str_path_generation_out_extero.c_str(), 0777);
                
                //set initial adaptive vector in executive area
                vector<vector<vector<double> > > dummy_mu(TEST_SEQ_NUM_LEARNING, vector<vector<double> >(TEST_MAX_LENGTH_LEARNING, vector<double>(executive_z_num, 0)));
                vector<vector<vector<double> > > dummy_sigma(TEST_SEQ_NUM_LEARNING, vector<vector<double> >(TEST_MAX_LENGTH_LEARNING, vector<double>(executive_z_num, 0)));
                for(int s=0;s<TEST_SEQ_NUM_LEARNING;++s){
                    stringstream str_a_mu, str_a_sigma;
                    str_a_mu << "./learning_model/" << "executive" << "/a_mu_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                    str_a_sigma << "./learning_model/" << "executive" << "/a_sigma_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                    set_weight(dummy_mu[s], str_a_mu.str());
                    set_weight(dummy_sigma[s], str_a_sigma.str());
                }
                set_initial_adaptive_vector(executive.a_mu, dummy_mu);
                set_initial_adaptive_vector(executive.a_sigma, dummy_sigma);
                
                //error regression
                for(int s=0;s<TEST_SEQ_NUM;++s){
                    //measure time
                    struct timeval start, end;
                    gettimeofday(&start, NULL);
                    int time_length = length[s];
                    for(int ct=0;ct<time_length;++ct){//ct: current time step
                        printf("Window size: %d, Epoch size: %d, Alpha: %lf, Sequence: %d, Time step: %d\n", window, iteration, alpha, s, ct);
                        int initial_regression;
                        (ct==0) ? (initial_regression=1) : (initial_regression=0);
                        if(ct>0){ //increment size of temporal variables
                            executive.er_increment_timestep(s);
                            associative.er_increment_timestep(s);
                            proprioceptive.er_increment_timestep(s);
                            exteroceptive.er_increment_timestep(s);
                            out_proprio.er_increment_timestep(s);
                            out_extero.er_increment_timestep(s);
                            target_proprio[s].emplace_back(vector<double>(x_proprio_num,0));
                            joint_angle[s].emplace_back(vector<double>(x_proprio_num,0));
                            target_extero[s].emplace_back(vector<double>(x_extero_num,0));
                            prop_exte_grad_internal_state_d[s].emplace_back(vector<double>(proprioceptive_d_num+exteroceptive_d_num,0));
                        }
                        int er_window;
                        (ct>=window-1) ? (er_window=window) : (er_window=ct+1);
                        for(int epoch=0;epoch<iteration;++epoch){
                            //Feedforward
                            for(int wt=ct-er_window+1;wt<=ct;++wt){//wt: time step within time window
                                int first_window_step, last_window_step;
                                (wt==ct-er_window+1) ? (first_window_step=1) : (first_window_step=0);
                                (wt==ct) ? (last_window_step=1) : (last_window_step=0);
                                executive.er_forward(wt, s, epoch, first_window_step, "posterior");
                                associative.er_forward(executive.z, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                //associative.er_forward(wt, s, epoch, initial_regression, last_window_step, "posterior");
                                proprioceptive.er_forward(associative.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                exteroceptive.er_forward(associative.d, wt, s, epoch, initial_regression, last_window_step, "posterior");
                                out_proprio.er_forward(proprioceptive.d, wt, s);
                                out_extero.er_forward(exteroceptive.d, wt, s);
                            }
                            //set incoming target at the current time step (if online robot operation)
                            //set through PID control using proprioceptive prediction in robot operation
                            if(epoch==0){
                                for(int i=0;i<x_num;++i){
                                    if(i<x_proprio_num){
                                        if(ct==0){
                                            joint_angle[s][ct][i] = PID(joint_angle[s][ct][i], out_proprio.output[s][ct][i]);
                                        }else{
                                            joint_angle[s][ct][i] = PID(joint_angle[s][ct-1][i], out_proprio.output[s][ct][i]);
                                        }
                                        target_proprio[s][ct][i] = joint_angle[s][ct][i];
                                    }else{
                                        if(ct<100){
                                            target_extero_dummy = fk(joint_angle[s][ct]);
                                            target_extero[s][ct][i-x_proprio_num] = target_extero_dummy[i-x_proprio_num];
                                        }else{
                                            target_extero[s][ct][i-x_proprio_num] = target[s][ct][i];
                                        }
                                    }
                                }         
                            }
                            
                            //Backward
                            //set constant for Rectified Adam
                            double rho_inf = 2.0/(1.0-ADAM2)-1.0;
                            double rho = rho_inf-2.0*(epoch+1)*pow(ADAM2,epoch+1)/(1.0-pow(ADAM2,epoch+1));
                            double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/ (rho_inf-4.0)/(rho_inf-2.0)/rho);
                            double m_radam1 = 1.0/(1.0-pow(ADAM1,epoch+1));
                            double l_radam2 = 1.0-pow(ADAM2,epoch+1);
                            for(int wt=ct;wt>=ct-er_window+1;--wt){
                                int first_window_step, last_window_step;
                                (wt==ct-er_window+1) ? (first_window_step=1) : (first_window_step=0);
                                (wt==ct) ? (last_window_step=1) : (last_window_step=0);
                                out_extero.er_backward(target_extero, wt, s);
                                out_proprio.er_backward(target_proprio, wt, s);
                                exteroceptive.er_backward(out_extero.weight_ohd, out_extero.grad_internal_state_output, out_extero.tau, last_window_step, wt, s);
                                proprioceptive.er_backward(out_proprio.weight_ohd, out_proprio.grad_internal_state_output, out_proprio.tau, last_window_step, wt, s);
                                //concatenate some variables in proprioceptive and exteroceptive areas to make backprop-inputs to the associative area
                                for(int i=0;i<proprioceptive_d_num+exteroceptive_d_num;++i){
                                    if(i<proprioceptive_d_num){
                                        prop_exte_grad_internal_state_d[s][wt][i] = proprioceptive.grad_internal_state_d[s][wt][i];
                                        prop_exte_tau[i] = proprioceptive.tau[i];
                                    }else{
                                        prop_exte_grad_internal_state_d[s][wt][i] = exteroceptive.grad_internal_state_d[s][wt][i-proprioceptive_d_num];
                                        prop_exte_tau[i] = exteroceptive.tau[i-proprioceptive_d_num];
                                    }
                                    for(int j=0;j<associative_d_num;++j){
                                        if(i<proprioceptive_d_num){
                                            prop_exte_weight_ld[i][j] = proprioceptive.weight_dhd[i][j];
                                        }else{
                                            prop_exte_weight_ld[i][j] = exteroceptive.weight_dhd[i-proprioceptive_d_num][j];
                                        }
                                    } 
                                }
                                associative.er_backward(prop_exte_weight_ld, prop_exte_grad_internal_state_d, prop_exte_tau, last_window_step, wt, s);
                                executive.er_backward(associative.weight_dhd, associative.grad_internal_state_d, associative.tau, last_window_step, wt, s);
                                
                                //update adaptive vector (Rectified Adam)
                                exteroceptive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                proprioceptive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                associative.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                if(wt==ct-window+1){
                                    executive.er_update_parameter_radam(wt, s, epoch, alpha, ADAM1, ADAM2, rho, r_store, m_radam1, l_radam2);
                                }
                            }
                        }
                    }
                    //Save generated sequence
                    executive.er_save_sequence(str_path_generation_executive, time_length, s, 0);
                    associative.er_save_sequence(str_path_generation_associative, time_length, s, 0);
                    proprioceptive.er_save_sequence(str_path_generation_proprioceptive, time_length, s, 0);
                    exteroceptive.er_save_sequence(str_path_generation_exteroceptive, time_length, s, 0);
                    out_proprio.er_save_sequence(str_path_generation_out_proprio, target_proprio, time_length, s, 0);
                    out_extero.er_save_sequence(str_path_generation_out_extero, target_extero, time_length, s, 0);
                    
                    gettimeofday(&end, NULL);
                    float delta = end.tv_sec  - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
                    printf("time %lf[s]\n", delta);
                }
            }
        }
    }
    return 0;
}
*/
