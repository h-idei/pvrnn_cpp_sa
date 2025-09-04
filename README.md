# pvrnn_cpp_sa
C++ code of predictive-coding-inspired variational recurrent neural network model (PV-RNN)

1. Tested environment
    -OS: Ubuntu (20.04)
    -gcc/g++ version 11.2.0

2. Installation guide
    This code requires only gcc/g++.

3. Demo and Instructions
    
The file structure is below.

“network.hpp”: Network hyperparameter setting, neural network classes, and supplemental functions (used in both learning and error regression).
“network.hpp” defines some classes like 
- "Output", "Stochastic_Output", "PVRNNLayer", "PVRNNTopLayer", and "PVRNNTopLayerLatent", which are used for learning
- "ER_Output", "ER_Stochastic_Output", "ER_PVRNNLayer", "ER_PVRNNTopLayer", and "ER_PVRNNTopLayerLatent", which are used for error regression

Codes for learning
“learning.cpp”: Main code for running program (learning)
“learning_data/”: Target data of learning is placed here (e.g., target0000000.txt ~ )
“learning_model/”: Trained model (e.g., synaptic weights) are saved here
“learning_generation/”: Reproduced sequences (learning results) are saved here
“plot_learning.py”: Code for generating figure

Codes for error regression
“error_regression.cpp”: Main code for running program (error regression)
“test_data/”: Target data of error regression is placed here (e.g., target0000000.txt ~ )
“test_generation/”: Reproduced sequences (results of error regression) are saved here
***Please prepare folders under which the results with each hyper-parameter setting (e.g., window size, iteration, and learning rate) are saved, like “test_generation/window_10/epoch_10/lr0.001/”, “test_generation/window_10/epoch_10/lr0.005/”,,,,. (The folder structure depends on the hyperparameter lists defined in “error_regression.cpp”)
“plot_er.py”: Code for generating figure

3-1. Learning experiment 
***It may take about 6294.689941[s] to train one neural network (200,000 epoch), with the same hyperparameter setting used in “Idei, H., Ohata, W., Yamashita, Y. et al. Emergence of sensory attenuation based upon the free-energy principle. Sci Rep 12, 14542 (2022). https://doi.org/10.1038/s41598-022-18207-7.”
    
    #Compilation and execution
    g++-11 -std=c++14 -O3 -fopenmp learning.cpp -o exe.learning
    ./exe.learning
    #Trained model and reproduced timeseries will be saved every 5000 learning epoch for each network module as 
    #"./learning_model/associative/weight_dd_*.txt" and "./learning_generation/out_extero/output_*_*.txt".
    
    #“plot_learning.py” can be used to generate figures (pdf files) of timeseries
    #saved in “./”.
    python plot_learning.py -s 0 -e 200000

3-2. Test experiment 
***one trial (200 time steps) by one neural network may take about 0.519920[s], with the same setting used in “Idei, H., Ohata, W., Yamashita, Y. et al. Emergence of sensory attenuation based upon the free-energy principle. Sci Rep 12, 14542 (2022). https://doi.org/10.1038/s41598-022-18207-7.”

    #Compilation and execution
    #“error_regression.cpp” reads a trained model from "./learning_model/" (in “network.hpp”, set learning epoch of a trained model which you want to read, like "string trained_model_index = "0100000";")
    #Test trial run
    g++-11 -std=c++14 -O3 error_regression.cpp -o exe.test
    ./exe.test
    #Timeseries for each trial will be saved in “test_generation/window_10/epoch_10/lr0.001/”, “test_generation/window_10/epoch_10/lr0.005/”,,,,. (The folder structure depends on the hyper-parameter lists in “error_regression.cpp”, and so on. 
    
    #“plot_er.py” can be used to generate figure (pdf file) of timeseries for each trial
    #saved in the same folder with generated timeseries
    python plot_er.py -s 0 -e 0
    

#####Additional Information#####
---Setting of initial adaptive vectors in error regression
By default, initial adaptive vectors in exteroceptive, proprioceptive, and associative areas are set with initial internal states of the corresponding prior.
Initial adaptive vectors in the executive area are set to the median values obtained for 24 training datasets of the self-produced context developed through learning.

***If you want to set initial adaptive vectors in the exteroceptive, proprioceptivr, and associative areas in the same way with the executive area, please 
1. comment out the following lines in “er_forward()” functions in “ER_PVRNNLayer” class (and “ER_PVRNNTopLayer” class, if used) in “network.hpp”.

    //initialization of posterior with prior
    if(epoch==0 && initial_regression==1){
        a_mu[s][t][i] = store1;
        a_sigma[s][t][i] = store2;
    }


2. add the processing in “error_regression.cpp” by referring to the following lines for the executive area,

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


***If you want to set initial adaptive vectors in the executive area in the same way with other areas, please 
1. comment out or remove the following lines in “error_regression.cpp”.

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


2. remove comment-out of the following part in “er_forward()” function in “ER_PVRNNTopLayerLatent” class in “network.hpp”

    //initialization of posterior with prior
    if(epoch==0 && t==0){//initialize by unit Gaussian prior N(0,1), or initial prior
        a_mu[s][t][i] = 0.0;
        a_sigma[s][t][i] = 0.0;
    }
