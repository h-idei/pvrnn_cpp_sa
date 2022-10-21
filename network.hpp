//  Predictive-coding-inspired Variational RNN
//  network.hpp
//  Copyright © 2022 Hayato Idei. All rights reserved.
//
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
using namespace std;
//network setting (the following setting is based on “Idei, H., Ohata, W., Yamashita, Y. et al. Emergence of sensory attenuation based upon the free-energy principle. Sci Rep 12, 14542 (2022). https://doi.org/10.1038/s41598-022-18207-7.”)
//executive level
#define executive_z_num 1 //the number of latent neurons
#define executive_W 0.005 //meta-prior
//associative level
#define associative_d_num 15 //the number of deterministic neurons
#define associative_z_num 3 //the number of latent neurons
#define associative_tau 2.0 //time constant of deterministic neurons
#define associative_W 0.005 //meta-prior
//sensory level
#define proprioceptive_d_num 15 //the number of deterministic neurons
#define proprioceptive_z_num 1 //the number of latent neurons
#define proprioceptive_tau 2.0 //time constant of deterministic neurons
#define proprioceptive_W 0.005 //meta-prior
#define exteroceptive_d_num 15 //the number of deterministic neurons
#define exteroceptive_z_num 1 //the number of latent neurons
#define exteroceptive_tau 2.0 //time constant of deterministic neurons
#define exteroceptive_W 0.005 //meta-prior
//output
#define x_num 5 //the dimension of all sensations, used for normalization
#define x_proprio_num 3 //the dimension of proprioception
#define x_extero_num 2 //the dimension of exteroception

#define ETA 0.0000001 //used to avoid null computation
#define PI 3.141592653589793
#define WEIGHT_SEED 1
mt19937 engine(WEIGHT_SEED);

//hyper-parameter used in error regression
string trained_model_index = "0100000"; //set learning epoch of trained model which you want to read

vector<string> split(string& input, char delimiter){
    istringstream stream(input);
    string field;
    vector<string> result;
    while (getline(stream, field, delimiter)) {
        result.push_back(field);
    }
    return result;
}

void set_target_data(vector<int>& length, vector<vector<vector<double> > >& target, string data_name){
    int sequence_size = int(target.size());
    vector<ifstream> ifs_target(target.size());
    for ( int s = 0; s < sequence_size; ++s ) {
        ostringstream ostr;
        ostr << "./" + data_name + "/target_" << setw(7) << setfill('0') << s << ".txt";
        ifs_target[s].open(ostr.str());
        if( ifs_target[s].fail()){
            cerr << "Failed to open file." << endl;
        }
        int t=0;
        string line;
        while (getline(ifs_target[s], line)) {
            vector<string> strvec = split(line, '\t');
            int index_size = int(strvec.size());
            for (int i=0; i<index_size ;++i){
                if(t < target[s].size()) target[s][t][i] = stod(strvec.at(i));
            }
            ++t;
        }
        length[s] = t;
    }    
}

void set_weight(vector<vector<double> >& weight, string filename = "Xavier"){
    if(filename == "Xavier"){
        int weight_size_1 = int(weight.size());
        int weight_size_2 = int(weight[0].size());
        double gain = 0.01;
        double K = gain * 2.0 / (weight_size_1+weight_size_2); //1.0/weight_size_2;
        double mean = 0.0, var = 0.0;
        normal_distribution<> dist(0.0, sqrt(K));
        int t=0;
        if(weight_size_1*weight_size_2==1){
            for(int i = 0;i < weight_size_1; ++i){
                for(int j = 0; j < weight_size_2; ++j){
                    weight[i][j] = dist(engine);
                }
            }
        }else{
            while(fabs(K - var) > 0.01 * K || fabs(mean) > 0.001){
                mean=0.0;
                var=0.0;
                for(int i = 0;i < weight_size_1; ++i){
                    for(int j = 0; j < weight_size_2; ++j){
                        weight[i][j] = dist(engine);
                        mean += weight[i][j]/(weight_size_1*weight_size_2);
                    }
                }
                for(int i = 0;i < weight_size_1; ++i){
                    for(int j = 0; j < weight_size_2; ++j){
                        var += pow(mean-weight[i][j],2)/(weight_size_1*weight_size_2);
                    }
                }   
            }
        }
    }else if(filename == "He"){
        int weight_size_1 = int(weight.size());
        int weight_size_2 = int(weight[0].size());
        double gain = 0.01;
        double K = gain * 2.0 / weight_size_2; // 4.0/(weight_size_2+weight_size_2)
        double mean = 0.0, var = 0.0;
        normal_distribution<> dist(0.0, sqrt(K));
        if(weight_size_1*weight_size_2==1){
            for(int i = 0;i < weight_size_1; ++i){
                for(int j = 0; j < weight_size_2; ++j){
                    weight[i][j] = dist(engine);
                }
            }
        }else{
            while(fabs(K - var) > 0.01 * K || fabs(mean) > 0.001){
                mean=0.0;
                var=0.0;
                for(int i = 0;i < weight_size_1; ++i){
                    for(int j = 0; j < weight_size_2; ++j){
                        weight[i][j] = dist(engine);
                        mean += weight[i][j]/(weight_size_1*weight_size_2);
                    }
                }
                for(int i = 0;i < weight_size_1; ++i){
                    for(int j = 0; j < weight_size_2; ++j){
                        var += pow(mean-weight[i][j],2)/(weight_size_1*weight_size_2);
                    }
                }   
            }
        }
    }else if(filename == "uniform"){
        int weight_size_1 = int(weight.size());
        int weight_size_2 = int(weight[0].size());
        double a = -sqrt(1.0/(weight_size_1+weight_size_2));//6.0/(weight_size_1+weight_size_2)
        double b = sqrt(1.0/(weight_size_1+weight_size_2));
        double K = pow(b-a,2)/12.0;
        double mean = 0.0, var = 0.0;
        uniform_real_distribution<> dist(a,b);
        if(weight_size_1*weight_size_2==1){
            for(int i = 0;i < weight_size_1; ++i){
                for(int j = 0; j < weight_size_2; ++j){
                    weight[i][j] = dist(engine);
                }
            }
        }else{
            while(fabs(K - var) > 0.01 * K || fabs(mean) > 0.001){
                mean=0.0;
                var=0.0;
                for(int i = 0;i < weight_size_1; ++i){
                    for(int j = 0; j < weight_size_2; ++j){
                        weight[i][j] = dist(engine);
                        mean += weight[i][j]/(weight_size_1*weight_size_2);
                    }
                }
                for(int i = 0;i < weight_size_1; ++i){
                    for(int j = 0; j < weight_size_2; ++j){
                        var += pow(mean-weight[i][j],2)/(weight_size_1*weight_size_2);
                    }
                }   
            }
        }
    }else{
        ifstream ifs(filename);
        if( ifs.fail()){
            cerr << "Failed to open file." << endl;
        }
        string line;
        int i=0;
        while (getline(ifs, line)) {
            vector<string> strvec = split(line, '\t');
            int index_size = int(strvec.size());
            for (int j=0; j<index_size ;++j){
                weight[i][j] = stod(strvec.at(j));
            }
            ++i;
        }
    }  
}

void set_bias(vector<double>& bias, string filename="normal"){
    int bias_size = int(bias.size());
    double K = 10.0;
    double mean = 0;
    double var = 0;
    if(filename == "normal"){
        normal_distribution<> dist(0.0, sqrt(K));
        while(fabs(K - var) > 0.01 * K && bias_size>1){
            for(int i = 0; i < bias_size; ++i){
                bias[i] = dist(engine);
            }
            mean = accumulate(begin(bias), end(bias), 0.0) / bias_size;
            var = accumulate(begin(bias), end(bias), 0.0, [mean](double sum, const auto& e){
                const auto temp = e - mean;
                return sum + temp * temp;
                }) / bias_size;
        }
    }else{
        ifstream ifs(filename);
        if( ifs.fail()){
            cerr << "Failed to open file." << endl;
        }
        string line;
        while (getline(ifs, line)) {
            vector<string> strvec = split(line, '\t');
            int index_size = int(strvec.size());
            for (int i=0; i<index_size ;++i){
                bias[i] = stod(strvec.at(i));
            }
        }
    }
}

void save_1dim(string& fo_path, string vector_name, vector<double>& vector, int epoch){
    int vector_size = int(vector.size());
    ostringstream ostr;
    ostr << fo_path << "/" << vector_name << "_" << setw(7) << setfill('0') << epoch << ".txt";
    ofstream ofs(ostr.str());
    for(int i=0; i < vector_size; ++i){
        ofs << vector[i];
        if (i != vector_size - 1) ofs << "\t";
    }
}

void save_2dim(string& fo_path, string vector_name, vector<vector<double> >& vector, int epoch){
    int vector_size_1 = int(vector.size());
    int vector_size_2 = int(vector[0].size());
    ostringstream ostr;
    ostr << fo_path << "/" << vector_name << "_" << setw(7) << setfill('0') << epoch << ".txt";
    ofstream ofs(ostr.str());
    for(int i=0; i < vector_size_1; ++i){
        for(int j = 0; j < vector_size_2; ++j){
            ofs << vector[i][j];
            if (j != vector_size_2 - 1) ofs << "\t";
        }
        if (i != vector_size_1 - 1) ofs << "\n";
    }
}

void save_generated_sequence(string& fo_path_generate, string output_name, vector<vector<vector<double> > >& output, int length, int s, int epoch){
    int output_size = int(output[0][0].size());
    ostringstream ostr;
    ostr << fo_path_generate << "/" << output_name << "_" << setw(7) << setfill('0') << s << "_"  << std::setw(7) << std::setfill('0') << epoch << ".txt";
    ofstream ofs(ostr.str());
    for(int i=0; i < length; ++i){
            for(int j = 0; j < output_size; ++j){
                ofs << output[s][i][j];
                if (j != output_size - 1) ofs << "\t";
            }
            if (i != length - 1) ofs << "\n";
        }
}

void set_initial_adaptive_vector(vector<vector<vector<double> > >& a, vector<vector<vector<double> > >& v){//set median value acquired for learned sequences
    int a_size_1 = int(a.size()); //number of test data
    int a_size_3 = int(a[0][0].size());
    int v_size = int(v.size()/2); //half number of training data (ex., self-produced context or externally produced context)
    double tmp;
    for(int i = 0; i < a_size_3; ++i){
        for(int s = 0; s < v_size - 1; ++s){
            for(int k = s + 1; k < v_size; ++k){
                if(v[s][0][i] > v[k][0][i]){
                    tmp = v[s][0][i];
                    v[s][0][i] = v[k][0][i];
                    v[k][0][i] = tmp;
                }
            }
        }
        //set median value to adaptive vector
        for(int s = 0; s < a_size_1; ++s){
            if (v_size % 2 == 1){
                a[s][0][i] = v[(v_size-1)/2][0][i];
            }else{
                a[s][0][i] = (v[(v_size/2)-1][0][i]+v[v_size/2][0][i])/2;
            }
        }
    }
}

//PID control 
double PID(double current, double goal){
    double joint = current, joint_pre = 0.0, error = 0.0, error_pre = 0.0, error_pre_pre = 0.0;
    int pid_step = 50; //number of iteration of pid
    vector<double> joint_trajectory(pid_step, 0);
    joint_trajectory[0] = current;
    //PID parameters
    double Kp = 0.3, Ki = 0.3, Kd = 0.02;
    for(int i = 1; i < pid_step; ++i){
        joint_pre = joint;
        error_pre_pre = error_pre;
        error_pre = error;
        error = goal - joint_trajectory[i-1]; //error = goal - previous joint angle;
        joint = joint_pre + Kp * (error-error_pre) + Ki * error + Kd * ((error-error_pre) - (error_pre-error_pre_pre));
        joint_trajectory[i] = joint;
    }
    return joint;
}

//forward kinematics
vector<double> fk(vector<double> joint){
    vector<double> hand_pos(2, 0);
    double l1 = 0.1, l2 = 0.3, l3 = 0.5;//lengths of links
    vector<double> th(3, 0);
    for(int i=0; i < 3; ++i){
        th[i] = (joint[i] + 0.8) * PI / 1.6;
    }
    //position of link 1
    double x1, y1;
    x1 = - (l1 * cos(th[0]));
    y1 = l1 * sin(th[0]);
    //position of link 2
    double x2, y2;
    x2 = x1 - (l2 * cos((th[0]+th[1])));
    y2 = y1 + (l2 * sin((th[0]+th[1])));
    //position of link 3
    double x3, y3;
    x3 = x2 - (l3 * cos((th[0]+th[1]+th[2])));
    y3 = y2 + (l3 * sin((th[0]+th[1]+th[2])));
    
    hand_pos[0] = x3;
    hand_pos[1] = y3;
    
    return hand_pos;
}

class Output{
private:
    int seq_size;
    int init_length;
    int out_size;
    int hd_size; //size of inputs from higher neurons
    string fo_path_parameter; //path to directory for saving trained model
    string fo_path_generation; //path to directory for saving generated sequence
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    //symaptic weights and related variables
    vector<vector<double> > weight_ohd;
    vector<vector<double> > grad_weight_ohd;
    vector<vector<vector<double> > > grad_weight_ohd_each_sequence;
    vector<vector<double> > adam_m_weight_ohd;
    vector<vector<double> > adam_v_weight_ohd;
    //time constant
    vector<double> tau;
    //dynamic variables
    vector<vector<vector<double> > > internal_state_output;
    vector<vector<vector<double> > > output;
    vector<vector<vector<double> > > grad_internal_state_output;
    vector<vector<vector<double> > > prediction_error;
    //initialize variables
    Output(int seq_num, int initial_length, int out_num, int hd_num, string& fo_path_param, string& fo_path_gen, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        out_size = out_num;
        hd_size = hd_num;
        if(fi_path_param=="learning"){
            fo_path_parameter = fo_path_param;
            mkdir(fo_path_param.c_str(), 0777);
        }
        fo_path_generation = fo_path_gen;
        mkdir(fo_path_gen.c_str(), 0777);
        fi_path_parameter = fi_path_param;
        weight_ohd.assign(out_num, vector<double>(hd_num, 0));
        grad_weight_ohd.assign(out_num, vector<double>(hd_num, 0));
        grad_weight_ohd_each_sequence.assign(seq_num, vector<vector<double> >(out_num, vector<double>(hd_num, 0)));
        adam_m_weight_ohd.assign(out_num, vector<double>(hd_num, 0));
        adam_v_weight_ohd.assign(out_num, vector<double>(hd_num, 0));
        tau.assign(out_num, 1); //used in backprop at higher level
        internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        grad_internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        prediction_error.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        //initialization
        if(fi_path_param=="learning"){
            set_weight(weight_ohd, "Xavier");
        }else{ //set trained model (example)
            stringstream str;
            str << "./learning_model/" << fi_path_param << "/weight_ohd_" << trained_model_index << ".txt";
            set_weight(weight_ohd, str.str());
        }
    }
    void forward(vector<vector<vector<double> > >& hd, int t, int s){ 
        //hd: input from higher neurons, t: time step, s: sequence
        double store1=0;
        for(int i=0;i<out_size;++i){
            store1=0;
            for(int j=0;j<hd_size;++j){
                store1 += weight_ohd[i][j] * hd[s][t][j];
            }
            internal_state_output[s][t][i] = store1;
            output[s][t][i] = tanh(store1);
        }
    }
    void backward(vector<vector<vector<double> > >& hd, vector<vector<vector<double> > >& target, int length, int t, int s){
        //compute gradient of internal state
        double normalize = 1.0*init_length/(length*out_size);
        for(int i=0;i<out_size;++i){
            prediction_error[s][t][i] = 0.5*pow(target[s][t][i]-output[s][t][i],2)*normalize;
            grad_internal_state_output[s][t][i] = -(1-pow(output[s][t][i],2))*(target[s][t][i]-output[s][t][i])*normalize;
            for(int j=0;j<hd_size;++j){
                grad_weight_ohd_each_sequence[s][i][j] += grad_internal_state_output[s][t][i]*hd[s][t][j]; //summation over time steps
            }
        }
    }
    void sum_gradient(int s){
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                grad_weight_ohd[i][j] += grad_weight_ohd_each_sequence[s][i][j];
            }
        }   
    }
    void update_parameter_radam(double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2, double weight_decay){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-adam2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(ADAM2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/(rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                adam_m_weight_ohd[i][j] = adam1*adam_m_weight_ohd[i][j]+(1-adam1)*grad_weight_ohd[i][j];
                adam_v_weight_ohd[i][j] = adam2*adam_v_weight_ohd[i][j]+(1-adam2)*pow(grad_weight_ohd[i][j],2);
                m_store = adam_m_weight_ohd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_ohd[i][j]+ETA));
                    weight_ohd[i][j] = weight_ohd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_ohd[i][j];
                }else{
                    weight_ohd[i][j] = weight_ohd[i][j]-alpha*m_store-weight_decay*weight_ohd[i][j];
                }
            }
        }
    }
    void update_parameter_adam(double alpha, double adam1, double adam2, double m_adam1, double v_adam2, double weight_decay){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                adam_m_weight_ohd[i][j] = adam1*adam_m_weight_ohd[i][j]+(1-adam1)*grad_weight_ohd[i][j];
                adam_v_weight_ohd[i][j] = adam2*adam_v_weight_ohd[i][j]+(1-adam2)*pow(grad_weight_ohd[i][j],2);
                m_store = adam_m_weight_ohd[i][j]*m_adam1;
                v_store = adam_v_weight_ohd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_ohd[i][j] = weight_ohd[i][j]-delta_store-weight_decay*weight_ohd[i][j];
            }
        }
    }
    void reset_gradient(){
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                grad_weight_ohd[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_ohd_each_sequence[s][i][j] = 0.0;
                }
            }
        }
    }
    void save_parameter(int epoch){
        save_2dim(fo_path_parameter, "weight_ohd", weight_ohd, epoch);
    }
    void save_sequence(vector<vector<vector<double> > >& target, int length, int s, int epoch){
        save_generated_sequence(fo_path_generation, "target", target, length, s, epoch);
        save_generated_sequence(fo_path_generation, "output", output, length, s, epoch);
        save_generated_sequence(fo_path_generation, "pe", prediction_error, length, s, epoch);
    }
};

class Stochastic_Output{
private:
    int seq_size;
    int init_length;
    int mean_size; //only mean or sigma
    int out_size; //mean+sigma
    int hd_size; //size of inputs from higher neurons
    string fo_path_parameter; //path to directory for saving trained model
    string fo_path_generation; //path to directory for saving generated sequence
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    //symaptic weights and related variables
    vector<vector<double> > weight_ohd;
    vector<vector<double> > grad_weight_ohd;
    vector<vector<vector<double> > > grad_weight_ohd_each_sequence;
    vector<vector<double> > adam_m_weight_ohd;
    vector<vector<double> > adam_v_weight_ohd;
    //time constant
    vector<double> tau;
    //dynamic variables
    vector<vector<vector<double> > > internal_state_output;
    vector<vector<vector<double> > > output;
    vector<vector<vector<double> > > grad_internal_state_output;
    vector<vector<vector<double> > > prediction_error; //negative log-likelihood
    //initialize variables
    Stochastic_Output(int seq_num, int initial_length, int out_num, int hd_num, string& fo_path_param, string& fo_path_gen, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        mean_size = out_num;
        out_size = 2*out_num;
        hd_size = hd_num;
        if(fi_path_param=="learning"){
            fo_path_parameter = fo_path_param;
            mkdir(fo_path_param.c_str(), 0777);
        }
        fo_path_generation = fo_path_gen;
        mkdir(fo_path_gen.c_str(), 0777);
        fi_path_parameter = fi_path_param;
        //mean:output[0~mean_size-1], sigma:output[mean_size~out_size-1]
        weight_ohd.assign(2*out_num, vector<double>(hd_num, 0));
        grad_weight_ohd.assign(2*out_num, vector<double>(hd_num, 0));
        grad_weight_ohd_each_sequence.assign(seq_num, vector<vector<double> >(2*out_num, vector<double>(hd_num, 0)));
        adam_m_weight_ohd.assign(2*out_num, vector<double>(hd_num, 0));
        adam_v_weight_ohd.assign(2*out_num, vector<double>(hd_num, 0));
        tau.assign(2*out_num, 1); //used in backprop at higher level
        internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(2*out_num, 0)));
        output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(2*out_num, 0)));
        grad_internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(2*out_num, 0)));
        prediction_error.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        //initialization
        if(fi_path_param=="learning"){
            set_weight(weight_ohd, "Xavier");
        }else{ //set trained model (example)
            stringstream str;
            str << "./learning_model/" << fi_path_param << "/weight_ohd_" << trained_model_index << ".txt";
            set_weight(weight_ohd, str.str());
        }
    }
    void forward(vector<vector<vector<double> > >& hd, int t, int s){ 
        //hd: input from higher neurons, t: time step, s: sequence
        double store1=0, store2=0; 
        for(int i=0;i<mean_size;++i){
            store1=0, store2=0;
            for(int j=0;j<hd_size;++j){
                store1 += weight_ohd[i][j] * hd[s][t][j];
                store2 += weight_ohd[i+mean_size][j] * hd[s][t][j];
            }
            internal_state_output[s][t][i] = store1;
            internal_state_output[s][t][i+mean_size] = store2;
            output[s][t][i] = tanh(store1);
            output[s][t][i+mean_size] = exp(store2)+ETA;
        }
    }
    void backward(vector<vector<vector<double> > >& hd, vector<vector<vector<double> > >& target, int length, int t, int s){
        //compute gradient of internal state
        //double normalize = init_length/length;
        double normalize = 1.0/(mean_size*length*seq_size);// Please note normalization method that is different from Output class
        for(int i=0;i<mean_size;++i){
            //negative log-likelihood
            prediction_error[s][t][i] = 0.5*(pow((target[s][t][i]-output[s][t][i])/output[s][t][i+mean_size],2)+log(2*PI)+2.0*log(output[s][t][i+mean_size]))*normalize;
            grad_internal_state_output[s][t][i] = -(1-pow(output[s][t][i],2))*(target[s][t][i]-output[s][t][i])/pow(output[s][t][i+mean_size],2)*normalize;
            grad_internal_state_output[s][t][i+mean_size] = (1.0-pow((target[s][t][i]-output[s][t][i])/output[s][t][i+mean_size],2))*normalize;
            for(int j=0;j<hd_size;++j){
                grad_weight_ohd_each_sequence[s][i][j] += grad_internal_state_output[s][t][i]*hd[s][t][j]; //summation over time steps
                grad_weight_ohd_each_sequence[s][i+mean_size][j] += grad_internal_state_output[s][t][i+mean_size]*hd[s][t][j]; //summation over time steps
            }
        }
    }
    void sum_gradient(int s){
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                grad_weight_ohd[i][j] += grad_weight_ohd_each_sequence[s][i][j];
            }
        }
    }
    void update_parameter_radam(double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2, double weight_decay){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-adam2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(adam2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/(rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                adam_m_weight_ohd[i][j] = adam1*adam_m_weight_ohd[i][j]+(1-adam1)*grad_weight_ohd[i][j];
                adam_v_weight_ohd[i][j] = adam2*adam_v_weight_ohd[i][j]+(1-adam2)*pow(grad_weight_ohd[i][j],2);
                m_store = adam_m_weight_ohd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_ohd[i][j]+ETA));
                    weight_ohd[i][j] = weight_ohd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_ohd[i][j];
                }else{
                    weight_ohd[i][j] = weight_ohd[i][j]-alpha*m_store-weight_decay*weight_ohd[i][j];
                }
            }
        }
    }
    void update_parameter_adam(double alpha, double adam1, double adam2, double m_adam1, double v_adam2, double weight_decay){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                adam_m_weight_ohd[i][j] = adam1*adam_m_weight_ohd[i][j]+(1-adam1)*grad_weight_ohd[i][j];
                adam_v_weight_ohd[i][j] = adam2*adam_v_weight_ohd[i][j]+(1-adam2)*pow(grad_weight_ohd[i][j],2);
                m_store = adam_m_weight_ohd[i][j]*m_adam1;
                v_store = adam_v_weight_ohd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_ohd[i][j] = weight_ohd[i][j]-delta_store-weight_decay*weight_ohd[i][j];
            }
        }
    }
    void reset_gradient(){
        for(int i=0;i<out_size;++i){
            for(int j=0;j<hd_size;++j){
                grad_weight_ohd[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_ohd_each_sequence[s][i][j] = 0.0;
                }
            }
        }
    }
    void save_parameter(int epoch){
        save_2dim(fo_path_parameter, "weight_ohd", weight_ohd, epoch);
    }
    void save_sequence(vector<vector<vector<double> > >& target, int length, int s, int epoch){
        save_generated_sequence(fo_path_generation, "target", target, length, s, epoch);
        save_generated_sequence(fo_path_generation, "output", output, length, s, epoch);
        save_generated_sequence(fo_path_generation, "pe", prediction_error, length, s, epoch);
    }
};

class PVRNNLayer{
private:
    int seq_size;
    int init_length;
    int d_size;
    int hd_size; //size of inputs from higher deterministic neurons
    int z_size;
    double W;
    string fo_path_parameter; //path to directory for saving trained model
    string fo_path_generation; //path to directory for saving generated sequence
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    vector<vector<double> > weight_pmd;
    vector<vector<double> > grad_weight_pmd;
    vector<vector<vector<double> > > grad_weight_pmd_each_sequence;
    vector<vector<double> > adam_m_weight_pmd;
    vector<vector<double> > adam_v_weight_pmd;
    //connections to generate prior sigma
    vector<vector<double> > weight_psd;
    vector<vector<double> > grad_weight_psd;
    vector<vector<vector<double> > > grad_weight_psd_each_sequence;
    vector<vector<double> > adam_m_weight_psd;
    vector<vector<double> > adam_v_weight_psd;
    //connections from latent states
    vector<vector<double> > weight_dz;
    vector<vector<double> > grad_weight_dz;
    vector<vector<vector<double> > > grad_weight_dz_each_sequence;
    vector<vector<double> > adam_m_weight_dz;
    vector<vector<double> > adam_v_weight_dz;
    //top-down connections
    vector<vector<double> > weight_dhd;
    vector<vector<double> > grad_weight_dhd;
    vector<vector<vector<double> > > grad_weight_dhd_each_sequence;
    vector<vector<double> > adam_m_weight_dhd;
    vector<vector<double> > adam_v_weight_dhd;
    //recurrent connections
    vector<vector<double> > weight_dd;
    vector<vector<double> > grad_weight_dd;
    vector<vector<vector<double> > > grad_weight_dd_each_sequence;
    vector<vector<double> > adam_m_weight_dd;
    vector<vector<double> > adam_v_weight_dd;
    //bias
    vector<double> bias;
    //vector<double> grad_bias;
    //vector<vector<double> > grad_bias_each_sequence;
    //vector<double> adam_m_bias;
    //vector<double> adam_v_bias;
    //time constant
    vector<double> tau;
    //dynamic variables and related variables
    //noise
    vector<vector<vector<double> > > eps;
    //prior
    vector<vector<vector<double> > > internal_state_p_mu;
    vector<vector<vector<double> > > p_mu;
    vector<vector<vector<double> > > grad_internal_state_p_mu;
    vector<vector<vector<double> > > internal_state_p_sigma;
    vector<vector<vector<double> > > p_sigma;
    vector<vector<vector<double> > > grad_internal_state_p_sigma;
    //adaptive vector (posterior)
    vector<vector<vector<double> > > a_mu;
    vector<vector<vector<double> > > q_mu;
    vector<vector<vector<double> > > grad_a_mu;
    vector<vector<vector<double> > > adam_m_a_mu;
    vector<vector<vector<double> > > adam_v_a_mu;
    vector<vector<vector<double> > > a_sigma;
    vector<vector<vector<double> > > q_sigma;
    vector<vector<vector<double> > > grad_a_sigma;
    vector<vector<vector<double> > > adam_m_a_sigma;
    vector<vector<vector<double> > > adam_v_a_sigma;
    //latent state
    vector<vector<vector<double> > > z;
    //deterministic state
    vector<vector<vector<double> > > internal_state_d;
    vector<vector<vector<double> > > d;
    vector<vector<vector<double> > > previous_internal_state_d; //if not 0, initial priors can be updated through optimization of synaptic weights 
    vector<vector<vector<double> > > previous_d;
    vector<vector<vector<double> > > grad_internal_state_d;
    //KLD
    vector<vector<vector<double> > > kld;
    vector<vector<vector<double> > > wkld;
    //initialize variables
    PVRNNLayer(int seq_num, int initial_length, int d_num, int hd_num, int z_num, double w, double time_constant, string& fo_path_param, string& fo_path_gen, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        d_size = d_num;
        hd_size = hd_num;
        z_size = z_num;
        W = w;
        if(fi_path_param=="learning"){
            fo_path_parameter = fo_path_param;
            mkdir(fo_path_param.c_str(), 0777);
        }
        fo_path_generation = fo_path_gen;
        mkdir(fo_path_gen.c_str(), 0777);
        fi_path_parameter = fi_path_param;
        weight_pmd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_pmd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_pmd_each_sequence.assign(seq_num, vector<vector<double> >(z_num, vector<double>(d_num, 0)));
        adam_m_weight_pmd.assign(z_num, vector<double>(d_num, 0));
        adam_v_weight_pmd.assign(z_num, vector<double>(d_num, 0));
        weight_psd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_psd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_psd_each_sequence.assign(seq_num, vector<vector<double> >(z_num, vector<double>(d_num, 0)));
        adam_m_weight_psd.assign(z_num, vector<double>(d_num, 0));
        adam_v_weight_psd.assign(z_num, vector<double>(d_num, 0));
        weight_dz.assign(d_num, vector<double>(z_num, 0));
        grad_weight_dz.assign(d_num, vector<double>(z_num, 0));
        grad_weight_dz_each_sequence.assign(seq_num, vector<vector<double> >(d_num, vector<double>(z_num, 0)));
        adam_m_weight_dz.assign(d_num, vector<double>(z_num, 0));
        adam_v_weight_dz.assign(d_num, vector<double>(z_num, 0));
        weight_dhd.assign(d_num, vector<double>(hd_num, 0));
        grad_weight_dhd.assign(d_num, vector<double>(hd_num, 0));
        grad_weight_dhd_each_sequence.assign(seq_num, vector<vector<double> >(d_num, vector<double>(hd_num, 0)));
        adam_m_weight_dhd.assign(d_num, vector<double>(hd_num, 0));
        adam_v_weight_dhd.assign(d_num, vector<double>(hd_num, 0));
        weight_dd.assign(d_num, vector<double>(d_num, 0));
        grad_weight_dd.assign(d_num, vector<double>(d_num, 0));
        grad_weight_dd_each_sequence.assign(seq_num, vector<vector<double> >(d_num, vector<double>(d_num, 0)));
        adam_m_weight_dd.assign(d_num, vector<double>(d_num, 0));
        adam_v_weight_dd.assign(d_num, vector<double>(d_num, 0));
        bias.assign(d_num, 0);
        //grad_bias.assign(d_num, 0);
        //grad_bias_each_sequence.assign(seq_num, vector<double>(d_num, 0));
        //adam_m_bias.assign(d_num, 0);
        //adam_v_bias.assign(d_num, 0);
        tau.assign(d_num, time_constant);
        for(int i=0;i<d_num;++i){
            if(i>=0.5*d_num){
                tau[i]=2*time_constant;
                }
        }
        eps.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        z.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        previous_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 1))); //if not 0, initial priors can be updated through optimization of synaptic weights 
        previous_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        grad_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        kld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        wkld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        //initialization
        if(fi_path_param=="learning"){
            set_weight(weight_pmd, "Xavier");
            set_weight(weight_psd, "Xavier");
            set_weight(weight_dz, "Xavier");
            set_weight(weight_dhd, "Xavier");
            set_weight(weight_dd, "Xavier");
            set_bias(bias, "normal");
        }else{ //set trained model (example)
            stringstream str_pmd, str_psd, str_dz, str_dhd, str_dd, str_bias;
            str_pmd << "./learning_model/" << fi_path_param << "/weight_pmd_" << trained_model_index << ".txt";
            str_psd << "./learning_model/" << fi_path_param << "/weight_psd_" << trained_model_index << ".txt";
            str_dz << "./learning_model/" << fi_path_param << "/weight_dz_" << trained_model_index << ".txt";
            str_dhd << "./learning_model/" << fi_path_param << "/weight_dhd_" << trained_model_index << ".txt";
            str_dd << "./learning_model/" << fi_path_param << "/weight_dd_" << trained_model_index << ".txt";
            str_bias << "./learning_model/" << fi_path_param << "/bias_" << trained_model_index << ".txt";
            set_weight(weight_pmd, str_pmd.str());
            set_weight(weight_psd, str_psd.str());
            set_weight(weight_dz, str_dz.str());
            set_weight(weight_dhd, str_dhd.str());
            set_weight(weight_dd, str_dd.str());
            set_bias(bias, str_bias.str());
            /*if you want to set initial adaptive vector externally (by default, initial adaptive vector was initialized with initial prior)
            vector<vector<vector<double> > > dummy_mu(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
            vector<vector<vector<double> > > dummy_sigma(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
            for(int s=0;s<seq_num;++s){
                stringstream str_a_mu, str_a_sigma;
                str_a_mu << "./learning_model/" << fi_path_param << "/a_mu_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                str_a_sigma << "./learning_model/" << fi_path_param << "/a_sigma_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                set_weight(dummy_mu[s], str_a_mu.str());
                set_weight(dummy_sigma[s], str_a_sigma.str());
            }
            set_initial_adaptive_vector(a_mu, dummy_mu);
            set_initial_adaptive_vector(a_sigma, dummy_sigma);
            */
        }
    }
    void set_eps(){
        normal_distribution<> dist(0, 1);
        for(int s=0;s<seq_size;++s){
            for(int t=0;t<init_length;++t){
                for(int i=0;i<z_size;++i){
                    eps[s][t][i] = dist(engine);
                }
            }
        }
    }
    void forward(vector<vector<vector<double> > >& hd, int t, int s, int epoch, string mode="posterior"){
        double store1=0, store2=0, store3=0;
        //set previous deterministic states
        for(int i=0;i<d_size;++i){
            if(t != 0) previous_internal_state_d[s][t][i] = internal_state_d[s][t-1][i];
            previous_d[s][t][i] = tanh(previous_internal_state_d[s][t][i]);
        }
        //compute latent states
        for(int i=0;i<z_size;++i){
            store1=0.0;
            store2=0.0;
            for(int j=0;j<d_size;++j){
                store1 += weight_pmd[i][j]*previous_d[s][t][j];
                store2 += weight_psd[i][j]*previous_d[s][t][j];
            }
            internal_state_p_mu[s][t][i] = store1;
            internal_state_p_sigma[s][t][i] = store2;
            p_mu[s][t][i] = tanh(store1);
            p_sigma[s][t][i] = exp(store2)+ETA;
            //initialize internal posterior state with initial internal prior state
            if(epoch==0){
                a_mu[s][t][i] = store1;
                a_sigma[s][t][i] = store2;
            }
            q_mu[s][t][i] = tanh(a_mu[s][t][i]);
            q_sigma[s][t][i] = exp(a_sigma[s][t][i])+ETA;
            if(mode=="posterior"){
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }else if(mode=="prior"){
                    z[s][t][i] = p_mu[s][t][i]+eps[s][t][i]*p_sigma[s][t][i];
            }else{//posterior generation
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }
        }
        //compute deterministic states
        for(int i=0;i<d_size;++i){
            //recurrent connections
            store1=0;
            for(int j=0;j<d_size;++j){
                store1 += weight_dd[i][j]*previous_d[s][t][j];
            }
            //top-down connections
            store2=0;
            for(int j=0;j<hd_size;++j){
                store2 += weight_dhd[i][j]*hd[s][t][j];
            }
            //connections from latent states
            store3=0;
            for(int j=0;j<z_size;++j){
                store3 += weight_dz[i][j]*z[s][t][j];
            }
            internal_state_d[s][t][i] = (1.0-1.0/tau[i])*previous_internal_state_d[s][t][i]+1.0/tau[i]*(store1+store2+store3+bias[i]);
            d[s][t][i] = tanh(internal_state_d[s][t][i]);
        }
    }
    void backward(vector<vector<vector<double> > >& hd, vector<vector<double> >& weight_ld, vector<vector<vector<double> > >& grad_internal_state_lower, vector<double>& tau_lower, int length, int t, int s){ //weight_ld: connections to lower neurons, grad_internal_state_lower: gradient of internal states of lower neurons
        int l_size = int(weight_ld.size());
        double store1=0, store2=0, store3=0, store4=0;
        double normalize = 1.0*init_length/(length*z_size);
        for(int i=0;i<d_size;++i){
            //compute gradient of internal state of deterministic neuron
            if(t==length-1){
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*store1;
            }else{
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                //error from next deterministic states
                store2=0;
                for(int k=0;k<d_size;++k){
                    store2 += 1.0/tau[k]*weight_dd[k][i]*grad_internal_state_d[s][t+1][k];
                }
                //error from next prior states
                store3=0, store4=0;
                for(int k=0;k<z_size;++k){
                    store3 += weight_pmd[k][i]*grad_internal_state_p_mu[s][t+1][k];
                    store4 += weight_psd[k][i]*grad_internal_state_p_sigma[s][t+1][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*(store1+store2+store3+store4)+(1.0-1.0/tau[i])*grad_internal_state_d[s][t+1][i];
            }
            //compute gradient of weights and bias
            //grad_bias_each_sequence[s][i] += 1.0/tau[i]*grad_internal_state_d[s][t][i];
            for(int j=0;j<hd_size;++j){
                grad_weight_dhd_each_sequence[s][i][j] += 1.0/tau[i]*grad_internal_state_d[s][t][i]*hd[s][t][j]; //summation over time steps
            }
            for(int j=0;j<z_size;++j){
                grad_weight_dz_each_sequence[s][i][j] += 1.0/tau[i]*grad_internal_state_d[s][t][i]*z[s][t][j]; //summation over time steps
            }
            for(int j=0;j<d_size;++j){
                grad_weight_dd_each_sequence[s][i][j] += 1.0/tau[i]*grad_internal_state_d[s][t][i]*previous_d[s][t][j]; //summation over time steps
            }
        }
        //compute gradient of parameters related with prior and posterior
        for(int i=0;i<z_size;++i){
            kld[s][t][i] = (log(p_sigma[s][t][i])-log(q_sigma[s][t][i])+0.5*(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2)-0.5)*normalize;
            wkld[s][t][i] = W*kld[s][t][i];
            store1=0;
            for(int k=0;k<d_size;++k){
                store1 += 1.0/tau[k]*weight_dz[k][i]*grad_internal_state_d[s][t][k]; //gradient of z
            }
            grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*(store1+W*(q_mu[s][t][i]-p_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize);
            grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+ W*(-1.0+pow(q_sigma[s][t][i],2)/pow(p_sigma[s][t][i],2))*normalize;
            grad_internal_state_p_mu[s][t][i] = (1.0-pow(p_mu[s][t][i],2))*W*(p_mu[s][t][i]-q_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize;
            grad_internal_state_p_sigma[s][t][i] = W*(1.0-(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2))*normalize;
            for(int k=0;k<d_size;++k){
                grad_weight_pmd_each_sequence[s][i][k] += grad_internal_state_p_mu[s][t][i]*previous_d[s][t][k]; //summation over time steps
                grad_weight_psd_each_sequence[s][i][k] += grad_internal_state_p_sigma[s][t][i]*previous_d[s][t][k]; //summation over time steps
            }
        }
    }
    void sum_gradient(int s){  //summation over sequences
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                grad_weight_pmd[i][j] += grad_weight_pmd_each_sequence[s][i][j];
                grad_weight_psd[i][j] += grad_weight_psd_each_sequence[s][i][j];
            }
        }
        for(int i=0;i<d_size;++i){
            //grad_bias[i] += grad_bias_each_sequence[s][i];
            for(int j=0;j<hd_size;++j){
                grad_weight_dhd[i][j] += grad_weight_dhd_each_sequence[s][i][j];
            }
            for(int j=0;j<d_size;++j){
                grad_weight_dd[i][j] += grad_weight_dd_each_sequence[s][i][j];
            }
            for(int j=0;j<z_size;++j){
                grad_weight_dz[i][j] += grad_weight_dz_each_sequence[s][i][j];
            }
        }
    }
    void update_parameter_radam(double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2, double weight_decay){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-adam2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(adam2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/(rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                adam_m_weight_pmd[i][j] = adam1*adam_m_weight_pmd[i][j]+(1-adam1)*grad_weight_pmd[i][j];
                adam_v_weight_pmd[i][j] = adam2*adam_v_weight_pmd[i][j]+(1-adam2)*pow(grad_weight_pmd[i][j],2);
                m_store = adam_m_weight_pmd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_pmd[i][j]+ETA));
                    weight_pmd[i][j] = weight_pmd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_pmd[i][j];
                }else{
                    weight_pmd[i][j] = weight_pmd[i][j]-alpha*m_store-weight_decay*weight_pmd[i][j];
                }
                adam_m_weight_psd[i][j] = adam1*adam_m_weight_psd[i][j]+(1-adam1)*grad_weight_psd[i][j];
                adam_v_weight_psd[i][j] = adam2*adam_v_weight_psd[i][j]+(1-adam2)*pow(grad_weight_psd[i][j],2);
                m_store = adam_m_weight_psd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_psd[i][j]+ETA));
                    weight_psd[i][j] = weight_psd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_psd[i][j];
                }else{
                    weight_psd[i][j] = weight_psd[i][j]-alpha*m_store-weight_decay*weight_psd[i][j];
                }
            }
        }
        for(int i=0;i<d_size;++i){
            /*
            adam_m_bias[i] = adam1*adam_m_bias[i]+(1-adam1)*grad_bias[i];
            adam_v_bias[i] = adam2*adam_v_bias[i]+(1-adam2)*pow(grad_bias[i],2);
            m_store = adam_m_bias[i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_bias[i]+ETA));
                bias[i] = bias[i]-alpha*r_store*m_store*l_store;
            }else{
                bias[i] = bias[i]-alpha*m_store;
            }
            */
            for(int j=0;j<hd_size;++j){
                adam_m_weight_dhd[i][j] = adam1*adam_m_weight_dhd[i][j]+(1-adam1)*grad_weight_dhd[i][j];
                adam_v_weight_dhd[i][j] = adam2*adam_v_weight_dhd[i][j]+(1-adam2)*pow(grad_weight_dhd[i][j],2);
                m_store = adam_m_weight_dhd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_dhd[i][j]+ETA));
                    weight_dhd[i][j] = weight_dhd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_dhd[i][j];
                }else{
                    weight_dhd[i][j] = weight_dhd[i][j]-alpha*m_store-weight_decay*weight_dhd[i][j];
                }
            }
            for(int j=0;j<d_size;++j){
                adam_m_weight_dd[i][j] = adam1*adam_m_weight_dd[i][j]+(1-adam1)*grad_weight_dd[i][j];
                adam_v_weight_dd[i][j] = adam2*adam_v_weight_dd[i][j]+(1-adam2)*pow(grad_weight_dd[i][j],2);
                m_store = adam_m_weight_dd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_dd[i][j]+ETA));
                    weight_dd[i][j] = weight_dd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_dd[i][j];
                }else{
                    weight_dd[i][j] = weight_dd[i][j]-alpha*m_store-weight_decay*weight_dd[i][j];
                }
            }
            for(int j=0;j<z_size;++j){
                adam_m_weight_dz[i][j] = adam1*adam_m_weight_dz[i][j]+(1-adam1)*grad_weight_dz[i][j];
                adam_v_weight_dz[i][j] = adam2*adam_v_weight_dz[i][j]+(1-adam2)*pow(grad_weight_dz[i][j],2);
                m_store = adam_m_weight_dz[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_dz[i][j]+ETA));
                    weight_dz[i][j] = weight_dz[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_dz[i][j];
                }else{
                    weight_dz[i][j] = weight_dz[i][j]-alpha*m_store-weight_decay*weight_dz[i][j];
                }
            }
        }
        for(int s=0;s<seq_size;++s){
            for(int t=0;t<init_length;++t){
                for(int i=0;i<z_size;++i){
                    adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
                    adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
                    m_store = adam_m_a_mu[s][t][i]*m_adam1;
                    if(rho>4){
                        l_store = sqrt(l_adam2/(adam_v_a_mu[s][t][i]+ETA));
                        a_mu[s][t][i] = a_mu[s][t][i]-alpha*r_store*m_store*l_store;
                    }else{
                        a_mu[s][t][i] = a_mu[s][t][i]-alpha*m_store;
                    }
                    adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
                    adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
                    m_store = adam_m_a_sigma[s][t][i]*m_adam1;
                    if(rho>4){
                        l_store = sqrt(l_adam2/(adam_v_a_sigma[s][t][i]+ETA));
                        a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*r_store*m_store*l_store;
                    }else{
                        a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*m_store;
                    }                 
                }
            }
        }
    }
    void update_parameter_adam(double alpha, double adam1, double adam2, double m_adam1, double v_adam2, double weight_decay){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                adam_m_weight_pmd[i][j] = adam1*adam_m_weight_pmd[i][j]+(1-adam1)*grad_weight_pmd[i][j];
                adam_v_weight_pmd[i][j] = adam2*adam_v_weight_pmd[i][j]+(1-adam2)*pow(grad_weight_pmd[i][j],2);
                m_store = adam_m_weight_pmd[i][j]*m_adam1;
                v_store = adam_v_weight_pmd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_pmd[i][j] = weight_pmd[i][j]-delta_store-weight_decay*weight_pmd[i][j];
                
                adam_m_weight_psd[i][j] = adam1*adam_m_weight_psd[i][j]+(1-adam1)*grad_weight_psd[i][j];
                adam_v_weight_psd[i][j] = adam2*adam_v_weight_psd[i][j]+(1-adam2)*pow(grad_weight_psd[i][j],2);
                m_store = adam_m_weight_psd[i][j]*m_adam1;
                v_store = adam_v_weight_psd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_psd[i][j] = weight_psd[i][j]-delta_store-weight_decay*weight_psd[i][j];
            }
        }
        for(int i=0;i<d_size;++i){
            /*
            adam_m_bias[i] = adam1*adam_m_bias[i]+(1-adam1)*grad_bias[i];
            adam_v_bias[i] = adam2*adam_v_bias[i]+(1-adam2)*pow(grad_bias[i],2);
            m_store = adam_m_bias[i]*m_adam1;
            v_store = adam_v_bias[i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            bias[i] = bias[i]-delta_store;
            */
            for(int j=0;j<hd_size;++j){
                adam_m_weight_dhd[i][j] = adam1*adam_m_weight_dhd[i][j]+(1-adam1)*grad_weight_dhd[i][j];
                adam_v_weight_dhd[i][j] = adam2*adam_v_weight_dhd[i][j]+(1-adam2)*pow(grad_weight_dhd[i][j],2);
                m_store = adam_m_weight_dhd[i][j]*m_adam1;
                v_store = adam_v_weight_dhd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_dhd[i][j] = weight_dhd[i][j]-delta_store-weight_decay*weight_dhd[i][j];
            }
            for(int j=0;j<d_size;++j){
                adam_m_weight_dd[i][j] = adam1*adam_m_weight_dd[i][j]+(1-adam1)*grad_weight_dd[i][j];
                adam_v_weight_dd[i][j] = adam2*adam_v_weight_dd[i][j]+(1-adam2)*pow(grad_weight_dd[i][j],2);
                m_store = adam_m_weight_dd[i][j]*m_adam1;
                v_store = adam_v_weight_dd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_dd[i][j] = weight_dd[i][j]-delta_store-weight_decay*weight_dd[i][j];
            }
            for(int j=0;j<z_size;++j){
                adam_m_weight_dz[i][j] = adam1*adam_m_weight_dz[i][j]+(1-adam1)*grad_weight_dz[i][j];
                adam_v_weight_dz[i][j] = adam2*adam_v_weight_dz[i][j]+(1-adam2)*pow(grad_weight_dz[i][j],2);
                m_store = adam_m_weight_dz[i][j]*m_adam1;
                v_store = adam_v_weight_dz[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_dz[i][j] = weight_dz[i][j]-delta_store-weight_decay*weight_dz[i][j];
            }
        }
        for(int s=0;s<seq_size;++s){
            for(int t=0;t<init_length;++t){
                for(int i=0;i<z_size;++i){
                    adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
                    adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
                    m_store = adam_m_a_mu[s][t][i]*m_adam1;
                    v_store = adam_v_a_mu[s][t][i]*v_adam2;
                    delta_store = alpha * m_store/(sqrt(v_store)+ETA);
                    a_mu[s][t][i] = a_mu[s][t][i]-delta_store; 
                    adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
                    adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
                    m_store = adam_m_a_sigma[s][t][i]*m_adam1;
                    v_store = adam_v_a_sigma[s][t][i]*v_adam2;
                    delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                    a_sigma[s][t][i] = a_sigma[s][t][i]-delta_store;                    
                }
            }
        }
    }
    void reset_gradient(){
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                grad_weight_pmd[i][j] = 0.0;
                grad_weight_psd[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_pmd_each_sequence[s][i][j] = 0.0;
                    grad_weight_psd_each_sequence[s][i][j] = 0.0;
                }
            }
        }
        for(int i=0;i<d_size;++i){
            //grad_bias[i] = 0.0;
            for(int j=0;j<hd_size;++j){
                grad_weight_dhd[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_dhd_each_sequence[s][i][j] = 0.0;
                    //grad_bias_each_sequence[s][i] = 0.0;
                }
            }
            for(int j=0;j<d_size;++j){
                grad_weight_dd[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_dd_each_sequence[s][i][j] = 0.0;
                }
            }
            for(int j=0;j<z_size;++j){
                grad_weight_dz[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_dz_each_sequence[s][i][j] = 0.0;
                }
            }
        }        
    }
    void save_parameter(int epoch){
        save_2dim(fo_path_parameter, "weight_pmd", weight_pmd, epoch);
        save_2dim(fo_path_parameter, "weight_psd", weight_psd, epoch);
        save_2dim(fo_path_parameter, "weight_dz", weight_dz, epoch);
        save_2dim(fo_path_parameter, "weight_dhd", weight_dhd, epoch);
        save_2dim(fo_path_parameter, "weight_dd", weight_dd, epoch);
        save_1dim(fo_path_parameter, "bias", bias, epoch);
    }
    void save_sequence(int length, int s, int epoch){
        save_generated_sequence(fo_path_generation, "in_p_mu", internal_state_p_mu, length, s, epoch);
        save_generated_sequence(fo_path_generation, "in_p_sigma", internal_state_p_sigma, length, s, epoch);
        if(fi_path_parameter=="learning"){
            save_generated_sequence(fo_path_parameter, "a_mu", a_mu, length, s, epoch);
            save_generated_sequence(fo_path_parameter, "a_sigma", a_sigma, length, s, epoch);
        }else{
            save_generated_sequence(fo_path_generation, "a_mu", a_mu, length, s, epoch);
            save_generated_sequence(fo_path_generation, "a_sigma", a_sigma, length, s, epoch);
        }
        save_generated_sequence(fo_path_generation, "z", z, length, s, epoch);
        save_generated_sequence(fo_path_generation, "in_d", internal_state_d, length, s, epoch);
        save_generated_sequence(fo_path_generation, "wkld", wkld, length, s, epoch);
    }
};

class PVRNNTopLayer{
private:
    int seq_size;
    int init_length;
    int d_size;
    int z_size;
    double W;
    string fo_path_parameter; //path to directory for saving trained model
    string fo_path_generation; //path to directory for saving generated sequence
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    vector<vector<double> > weight_pmd;
    vector<vector<double> > grad_weight_pmd;
    vector<vector<vector<double> > > grad_weight_pmd_each_sequence;
    vector<vector<double> > adam_m_weight_pmd;
    vector<vector<double> > adam_v_weight_pmd;
    //connections to generate prior sigma
    vector<vector<double> > weight_psd;
    vector<vector<double> > grad_weight_psd;
    vector<vector<vector<double> > > grad_weight_psd_each_sequence;
    vector<vector<double> > adam_m_weight_psd;
    vector<vector<double> > adam_v_weight_psd;
    //connections from latent states
    vector<vector<double> > weight_dz;
    vector<vector<double> > grad_weight_dz;
    vector<vector<vector<double> > > grad_weight_dz_each_sequence;
    vector<vector<double> > adam_m_weight_dz;
    vector<vector<double> > adam_v_weight_dz;
    //recurrent connections
    vector<vector<double> > weight_dd;
    vector<vector<double> > grad_weight_dd;
    vector<vector<vector<double> > > grad_weight_dd_each_sequence;
    vector<vector<double> > adam_m_weight_dd;
    vector<vector<double> > adam_v_weight_dd;
    //bias
    vector<double> bias;
    //vector<double> grad_bias;
    //vector<vector<double> > grad_bias_each_sequence;
    //vector<double> adam_m_bias;
    //vector<double> adam_v_bias;
    //time constant
    vector<double> tau;
    //dynamic variables and related variables
    //noise
    vector<vector<vector<double> > > eps;
    //prior
    vector<vector<vector<double> > > internal_state_p_mu;
    vector<vector<vector<double> > > p_mu;
    vector<vector<vector<double> > > grad_internal_state_p_mu;
    vector<vector<vector<double> > > internal_state_p_sigma;
    vector<vector<vector<double> > > p_sigma;
    vector<vector<vector<double> > > grad_internal_state_p_sigma;
    //adaptive vector (posterior)
    vector<vector<vector<double> > > a_mu;
    vector<vector<vector<double> > > q_mu;
    vector<vector<vector<double> > > grad_a_mu;
    vector<vector<vector<double> > > adam_m_a_mu;
    vector<vector<vector<double> > > adam_v_a_mu;
    vector<vector<vector<double> > > a_sigma;
    vector<vector<vector<double> > > q_sigma;
    vector<vector<vector<double> > > grad_a_sigma;
    vector<vector<vector<double> > > adam_m_a_sigma;
    vector<vector<vector<double> > > adam_v_a_sigma;
    //latent state
    vector<vector<vector<double> > > z;
    //deterministic state
    vector<vector<vector<double> > > internal_state_d;
    vector<vector<vector<double> > > d;
    vector<vector<vector<double> > > previous_internal_state_d; //if not 0, initial priors can be updated through optimization of synaptic weights 
    vector<vector<vector<double> > > previous_d;
    vector<vector<vector<double> > > grad_internal_state_d;
    //KLD
    vector<vector<vector<double> > > kld;
    vector<vector<vector<double> > > wkld;
    //initialize variables
    PVRNNTopLayer(int seq_num, int initial_length, int d_num, int z_num, double w, double time_constant, string& fo_path_param, string& fo_path_gen, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        d_size = d_num;
        z_size = z_num;
        W = w;
        if(fi_path_param=="learning"){
            fo_path_parameter = fo_path_param;
            mkdir(fo_path_param.c_str(), 0777);
        }
        fo_path_generation = fo_path_gen;
        mkdir(fo_path_gen.c_str(), 0777);
        fi_path_parameter = fi_path_param;
        weight_pmd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_pmd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_pmd_each_sequence.assign(seq_num, vector<vector<double> >(z_num, vector<double>(d_num, 0)));
        adam_m_weight_pmd.assign(z_num, vector<double>(d_num, 0));
        adam_v_weight_pmd.assign(z_num, vector<double>(d_num, 0));
        weight_psd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_psd.assign(z_num, vector<double>(d_num, 0));
        grad_weight_psd_each_sequence.assign(seq_num, vector<vector<double> >(z_num, vector<double>(d_num, 0)));
        adam_m_weight_psd.assign(z_num, vector<double>(d_num, 0));
        adam_v_weight_psd.assign(z_num, vector<double>(d_num, 0));
        weight_dz.assign(d_num, vector<double>(z_num, 0));
        grad_weight_dz.assign(d_num, vector<double>(z_num, 0));
        grad_weight_dz_each_sequence.assign(seq_num, vector<vector<double> >(d_num, vector<double>(z_num, 0)));
        adam_m_weight_dz.assign(d_num, vector<double>(z_num, 0));
        adam_v_weight_dz.assign(d_num, vector<double>(z_num, 0));
        weight_dd.assign(d_num, vector<double>(d_num, 0));
        grad_weight_dd.assign(d_num, vector<double>(d_num, 0));
        grad_weight_dd_each_sequence.assign(seq_num, vector<vector<double> >(d_num, vector<double>(d_num, 0)));
        adam_m_weight_dd.assign(d_num, vector<double>(d_num, 0));
        adam_v_weight_dd.assign(d_num, vector<double>(d_num, 0));
        bias.assign(d_num, 0);
        //grad_bias.assign(d_num, 0);
        //grad_bias_each_sequence.assign(seq_num, vector<double>(d_num, 0));
        //adam_m_bias.assign(d_num, 0);
        //adam_v_bias.assign(d_num, 0);
        tau.assign(d_num, time_constant);
        for(int i=0;i<d_num;++i){
            if(i>=0.5*d_num){
                tau[i]=2*time_constant;
                }
        }
        eps.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        z.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        previous_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 1))); //if not 0, initial priors can be updated through optimization of synaptic weights 
        previous_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        grad_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        kld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        wkld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        //initialization
        if(fi_path_param=="learning"){
            set_weight(weight_pmd, "Xavier");
            set_weight(weight_psd, "Xavier");
            set_weight(weight_dz, "Xavier");
            set_weight(weight_dd, "Xavier");
            set_bias(bias, "normal");
        }else{ //set trained model (example)
            stringstream str_pmd, str_psd, str_dz, str_dd, str_bias;
            str_pmd << "./learning_model/" << fi_path_param << "/weight_pmd_" << trained_model_index << ".txt";
            str_psd << "./learning_model/" << fi_path_param << "/weight_psd_" << trained_model_index << ".txt";
            str_dz << "./learning_model/" << fi_path_param << "/weight_dz_" << trained_model_index << ".txt";
            str_dd << "./learning_model/" << fi_path_param << "/weight_dd_" << trained_model_index << ".txt";
            str_bias << "./learning_model/" << fi_path_param << "/bias_" << trained_model_index << ".txt";
            set_weight(weight_pmd, str_pmd.str());
            set_weight(weight_psd, str_psd.str());
            set_weight(weight_dz, str_dz.str());
            set_weight(weight_dd, str_dd.str());
            set_bias(bias, str_bias.str());
            /*if you want to set initial adaptive vector externally (by default, initial adaptive vector was initialized with initial prior)
            vector<vector<vector<double> > > dummy_mu(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
            vector<vector<vector<double> > > dummy_sigma(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
            for(int s=0;s<seq_num;++s){
                stringstream str_a_mu, str_a_sigma;
                str_a_mu << "./learning_model/" << fi_path_param << "/a_mu_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                str_a_sigma << "./learning_model/" << fi_path_param << "/a_sigma_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                set_weight(dummy_mu[s], str_a_mu.str());
                set_weight(dummy_sigma[s], str_a_sigma.str());
            }
            set_initial_adaptive_vector(a_mu, dummy_mu);
            set_initial_adaptive_vector(a_sigma, dummy_sigma);
            */
        }
    }
    void set_eps(){
        normal_distribution<> dist(0, 1);
        for(int s=0;s<seq_size;++s){
            for(int t=0;t<init_length;++t){
                for(int i=0;i<z_size;++i){
                    eps[s][t][i] = dist(engine);
                }
            }
        }
    }
    void forward(int t, int s, int epoch, string mode="posterior"){
        double store1=0, store2=0, store3=0;
        //set previous deterministic states
        for(int i=0;i<d_size;++i){
            if(t != 0) previous_internal_state_d[s][t][i] = internal_state_d[s][t-1][i];
            previous_d[s][t][i] = tanh(previous_internal_state_d[s][t][i]);
        }
        //compute latent states
        for(int i=0;i<z_size;++i){
            store1=0.0;
            store2=0.0;
            for(int j=0;j<d_size;++j){
                store1 += weight_pmd[i][j]*previous_d[s][t][j];
                store2 += weight_psd[i][j]*previous_d[s][t][j];
            }
            internal_state_p_mu[s][t][i] = store1;
            internal_state_p_sigma[s][t][i] = store2;
            p_mu[s][t][i] = tanh(store1);
            p_sigma[s][t][i] = exp(store2)+ETA;
            //initialize internal posterior state with initial internal prior state
            if(epoch==0){
                a_mu[s][t][i] = store1;
                a_sigma[s][t][i] = store2;
            }
            q_mu[s][t][i] = tanh(a_mu[s][t][i]);
            q_sigma[s][t][i] = exp(a_sigma[s][t][i])+ETA;
            if(mode=="posterior"){
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }else if(mode=="prior"){
                    z[s][t][i] = p_mu[s][t][i]+eps[s][t][i]*p_sigma[s][t][i];
            }else{//posterior generation
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }
        }
        //compute deterministic states
        for(int i=0;i<d_size;++i){
            //recurrent connections
            store1=0;
            for(int j=0;j<d_size;++j){
                store1 += weight_dd[i][j]*previous_d[s][t][j];
            }
            //connections from latent states
            store3=0;
            for(int j=0;j<z_size;++j){
                store3 += weight_dz[i][j]*z[s][t][j];
            }
            internal_state_d[s][t][i] = (1.0-1.0/tau[i])*previous_internal_state_d[s][t][i]+1.0/tau[i]*(store1+store3+bias[i]);
            d[s][t][i] = tanh(internal_state_d[s][t][i]);
        }
    }
    void backward(vector<vector<double> >& weight_ld, vector<vector<vector<double> > >& grad_internal_state_lower, vector<double>& tau_lower, int length, int t, int s){ //weight_ld: connections to lower neurons, grad_internal_state_lower: gradient of internal states of lower neurons
        int l_size = int(weight_ld.size());
        double store1=0, store2=0, store3=0, store4=0;
        double normalize = 1.0*init_length/(length*z_size);
        for(int i=0;i<d_size;++i){
            //compute gradient of internal state of deterministic neuron
            if(t==length-1){
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*store1;
            }else{
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                //error from next deterministic states
                store2=0;
                for(int k=0;k<d_size;++k){
                    store2 += 1.0/tau[k]*weight_dd[k][i]*grad_internal_state_d[s][t+1][k];
                }
                //error from next prior states
                store3=0, store4=0;
                for(int k=0;k<z_size;++k){
                    store3 += weight_pmd[k][i]*grad_internal_state_p_mu[s][t+1][k];
                    store4 += weight_psd[k][i]*grad_internal_state_p_sigma[s][t+1][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*(store1+store2+store3+store4)+(1.0-1.0/tau[i])*grad_internal_state_d[s][t+1][i];
            }
            //compute gradient of weights and bias
            //grad_bias_each_sequence[s][i] += 1.0/tau[i]*grad_internal_state_d[s][t][i];
            for(int j=0;j<z_size;++j){
                grad_weight_dz_each_sequence[s][i][j] += 1.0/tau[i]*grad_internal_state_d[s][t][i]*z[s][t][j]; //summation over time steps
            }
            for(int j=0;j<d_size;++j){
                grad_weight_dd_each_sequence[s][i][j] += 1.0/tau[i]*grad_internal_state_d[s][t][i]*previous_d[s][t][j]; //summation over time steps
            }
        }
        //compute gradient of parameters related with prior and posterior
        for(int i=0;i<z_size;++i){
            kld[s][t][i] = (log(p_sigma[s][t][i])-log(q_sigma[s][t][i])+0.5*(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2)-0.5)*normalize;
            wkld[s][t][i] = W*kld[s][t][i];
            store1=0;
            for(int k=0;k<d_size;++k){
                store1 += 1.0/tau[k]*weight_dz[k][i]*grad_internal_state_d[s][t][k]; //gradient of z
            }
            grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*(store1+W*(q_mu[s][t][i]-p_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize);
            grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+ W*(-1.0+pow(q_sigma[s][t][i],2)/pow(p_sigma[s][t][i],2))*normalize;
            grad_internal_state_p_mu[s][t][i] = (1.0-pow(p_mu[s][t][i],2))*W*(p_mu[s][t][i]-q_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize;
            grad_internal_state_p_sigma[s][t][i] = W*(1.0-(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2))*normalize;
            for(int k=0;k<d_size;++k){
                grad_weight_pmd_each_sequence[s][i][k] += grad_internal_state_p_mu[s][t][i]*previous_d[s][t][k]; //summation over time steps
                grad_weight_psd_each_sequence[s][i][k] += grad_internal_state_p_sigma[s][t][i]*previous_d[s][t][k]; //summation over time steps
            }
        }
    }
    void sum_gradient(int s){  //summation over sequences
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                grad_weight_pmd[i][j] += grad_weight_pmd_each_sequence[s][i][j];
                grad_weight_psd[i][j] += grad_weight_psd_each_sequence[s][i][j];
            }
        }
        for(int i=0;i<d_size;++i){
            //grad_bias[i] += grad_bias_each_sequence[s][i];
            for(int j=0;j<d_size;++j){
                grad_weight_dd[i][j] += grad_weight_dd_each_sequence[s][i][j];
            }
            for(int j=0;j<z_size;++j){
                grad_weight_dz[i][j] += grad_weight_dz_each_sequence[s][i][j];
            }
        }
    }
    void update_parameter_radam(double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2, double weight_decay){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-ADAM2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(adam2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/(rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                adam_m_weight_pmd[i][j] = adam1*adam_m_weight_pmd[i][j]+(1-adam1)*grad_weight_pmd[i][j];
                adam_v_weight_pmd[i][j] = adam2*adam_v_weight_pmd[i][j]+(1-adam2)*pow(grad_weight_pmd[i][j],2);
                m_store = adam_m_weight_pmd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_pmd[i][j]+ETA));
                    weight_pmd[i][j] = weight_pmd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_pmd[i][j];
                }else{
                    weight_pmd[i][j] = weight_pmd[i][j]-alpha*m_store-weight_decay*weight_pmd[i][j];
                }
                adam_m_weight_psd[i][j] = adam1*adam_m_weight_psd[i][j]+(1-adam1)*grad_weight_psd[i][j];
                adam_v_weight_psd[i][j] = adam2*adam_v_weight_psd[i][j]+(1-adam2)*pow(grad_weight_psd[i][j],2);
                m_store = adam_m_weight_psd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_psd[i][j]+ETA));
                    weight_psd[i][j] = weight_psd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_psd[i][j];
                }else{
                    weight_psd[i][j] = weight_psd[i][j]-alpha*m_store-weight_decay*weight_psd[i][j];
                }
            }
        }
        for(int i=0;i<d_size;++i){
            /*
            adam_m_bias[i] = adam1*adam_m_bias[i]+(1-adam1)*grad_bias[i];
            adam_v_bias[i] = adam2*adam_v_bias[i]+(1-adam2)*pow(grad_bias[i],2);
            m_store = adam_m_bias[i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_bias[i]+ETA));
                bias[i] = bias[i]-alpha*r_store*m_store*l_store;
            }else{
                bias[i] = bias[i]-alpha*m_store;
            }
            */
            for(int j=0;j<d_size;++j){
                adam_m_weight_dd[i][j] = adam1*adam_m_weight_dd[i][j]+(1-adam1)*grad_weight_dd[i][j];
                adam_v_weight_dd[i][j] = adam2*adam_v_weight_dd[i][j]+(1-adam2)*pow(grad_weight_dd[i][j],2);
                m_store = adam_m_weight_dd[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_dd[i][j]+ETA));
                    weight_dd[i][j] = weight_dd[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_dd[i][j];
                }else{
                    weight_dd[i][j] = weight_dd[i][j]-alpha*m_store-weight_decay*weight_dd[i][j];
                }
            }
            for(int j=0;j<z_size;++j){
                adam_m_weight_dz[i][j] = adam1*adam_m_weight_dz[i][j]+(1-adam1)*grad_weight_dz[i][j];
                adam_v_weight_dz[i][j] = adam2*adam_v_weight_dz[i][j]+(1-adam2)*pow(grad_weight_dz[i][j],2);
                m_store = adam_m_weight_dz[i][j]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_weight_dz[i][j]+ETA));
                    weight_dz[i][j] = weight_dz[i][j]-alpha*r_store*m_store*l_store-weight_decay*weight_dz[i][j];
                }else{
                    weight_dz[i][j] = weight_dz[i][j]-alpha*m_store-weight_decay*weight_dz[i][j];
                }
            }
        }
        for(int s=0;s<seq_size;++s){
            for(int t=0;t<init_length;++t){
                for(int i=0;i<z_size;++i){
                    adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
                    adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
                    m_store = adam_m_a_mu[s][t][i]*m_adam1;
                    if(rho>4){
                        l_store = sqrt(l_adam2/(adam_v_a_mu[s][t][i]+ETA));
                        a_mu[s][t][i] = a_mu[s][t][i]-alpha*r_store*m_store*l_store;
                    }else{
                        a_mu[s][t][i] = a_mu[s][t][i]-alpha*m_store;
                    }
                    adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
                    adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
                    m_store = adam_m_a_sigma[s][t][i]*m_adam1;
                    if(rho>4){
                        l_store = sqrt(l_adam2/(adam_v_a_sigma[s][t][i]+ETA));
                        a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*r_store*m_store*l_store;
                    }else{
                        a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*m_store;
                    }                 
                }
            }
        }
    }
    void update_parameter_adam(double alpha, double adam1, double adam2, double m_adam1, double v_adam2, double weight_decay){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                adam_m_weight_pmd[i][j] = adam1*adam_m_weight_pmd[i][j]+(1-adam1)*grad_weight_pmd[i][j];
                adam_v_weight_pmd[i][j] = adam2*adam_v_weight_pmd[i][j]+(1-adam2)*pow(grad_weight_pmd[i][j],2);
                m_store = adam_m_weight_pmd[i][j]*m_adam1;
                v_store = adam_v_weight_pmd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_pmd[i][j] = weight_pmd[i][j]-delta_store-weight_decay*weight_pmd[i][j];
                
                adam_m_weight_psd[i][j] = adam1*adam_m_weight_psd[i][j]+(1-adam1)*grad_weight_psd[i][j];
                adam_v_weight_psd[i][j] = adam2*adam_v_weight_psd[i][j]+(1-adam2)*pow(grad_weight_psd[i][j],2);
                m_store = adam_m_weight_psd[i][j]*m_adam1;
                v_store = adam_v_weight_psd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_psd[i][j] = weight_psd[i][j]-delta_store-weight_decay*weight_psd[i][j];
            }
        }
        for(int i=0;i<d_size;++i){
            /*
            adam_m_bias[i] = adam1*adam_m_bias[i]+(1-adam1)*grad_bias[i];
            adam_v_bias[i] = adam2*adam_v_bias[i]+(1-adam2)*pow(grad_bias[i],2);
            m_store = adam_m_bias[i]*m_adam1;
            v_store = adam_v_bias[i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            bias[i] = bias[i]-delta_store;
            */
            for(int j=0;j<d_size;++j){
                adam_m_weight_dd[i][j] = adam1*adam_m_weight_dd[i][j]+(1-adam1)*grad_weight_dd[i][j];
                adam_v_weight_dd[i][j] = adam2*adam_v_weight_dd[i][j]+(1-adam2)*pow(grad_weight_dd[i][j],2);
                m_store = adam_m_weight_dd[i][j]*m_adam1;
                v_store = adam_v_weight_dd[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_dd[i][j] = weight_dd[i][j]-delta_store-weight_decay*weight_dd[i][j];
            }
            for(int j=0;j<z_size;++j){
                adam_m_weight_dz[i][j] = adam1*adam_m_weight_dz[i][j]+(1-adam1)*grad_weight_dz[i][j];
                adam_v_weight_dz[i][j] = adam2*adam_v_weight_dz[i][j]+(1-adam2)*pow(grad_weight_dz[i][j],2);
                m_store = adam_m_weight_dz[i][j]*m_adam1;
                v_store = adam_v_weight_dz[i][j]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                weight_dz[i][j] = weight_dz[i][j]-delta_store-weight_decay*weight_dz[i][j];
            }
        }
        for(int s=0;s<seq_size;++s){
            for(int t=0;t<init_length;++t){
                for(int i=0;i<z_size;++i){
                    adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
                    adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
                    m_store = adam_m_a_mu[s][t][i]*m_adam1;
                    v_store = adam_v_a_mu[s][t][i]*v_adam2;
                    delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                    a_mu[s][t][i] = a_mu[s][t][i]-delta_store; 
                    adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
                    adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
                    m_store = adam_m_a_sigma[s][t][i]*m_adam1;
                    v_store = adam_v_a_sigma[s][t][i]*v_adam2;
                    delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                    a_sigma[s][t][i] = a_sigma[s][t][i]-delta_store;                    
                }
            }
        }
    }
    void reset_gradient(){
        for(int i=0;i<z_size;++i){
            for(int j=0;j<d_size;++j){
                grad_weight_pmd[i][j] = 0.0;
                grad_weight_psd[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_pmd_each_sequence[s][i][j] = 0.0;
                    grad_weight_psd_each_sequence[s][i][j] = 0.0;
                }
            }
        }
        for(int i=0;i<d_size;++i){
            //grad_bias[i] = 0.0;
            for(int j=0;j<d_size;++j){
                grad_weight_dd[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_dd_each_sequence[s][i][j] = 0.0;
                    //grad_bias_each_sequence[s][i] = 0.0;
                }
            }
            for(int j=0;j<z_size;++j){
                grad_weight_dz[i][j] = 0.0;
                for(int s=0;s<seq_size;++s){
                    grad_weight_dz_each_sequence[s][i][j] = 0.0;
                }
            }
        }        
    }
    void save_parameter(int epoch){
        save_2dim(fo_path_parameter, "weight_pmd", weight_pmd, epoch);
        save_2dim(fo_path_parameter, "weight_psd", weight_psd, epoch);
        save_2dim(fo_path_parameter, "weight_dz", weight_dz, epoch);
        save_2dim(fo_path_parameter, "weight_dd", weight_dd, epoch);
        save_1dim(fo_path_parameter, "bias", bias, epoch);
    }
    void save_sequence(int length, int s, int epoch){
        save_generated_sequence(fo_path_generation, "in_p_mu", internal_state_p_mu, length, s, epoch);
        save_generated_sequence(fo_path_generation, "in_p_sigma", internal_state_p_sigma, length, s, epoch);
        if(fi_path_parameter=="learning"){
            save_generated_sequence(fo_path_parameter, "a_mu", a_mu, length, s, epoch);
            save_generated_sequence(fo_path_parameter, "a_sigma", a_sigma, length, s, epoch);
        }else{
            save_generated_sequence(fo_path_generation, "a_mu", a_mu, length, s, epoch);
            save_generated_sequence(fo_path_generation, "a_sigma", a_sigma, length, s, epoch);
        }
        save_generated_sequence(fo_path_generation, "z", z, length, s, epoch);
        save_generated_sequence(fo_path_generation, "in_d", internal_state_d, length, s, epoch);
        save_generated_sequence(fo_path_generation, "wkld", wkld, length, s, epoch);
    }
};

class PVRNNTopLayerLatent{
private:
    int seq_size;
    int init_length;
    int z_size;
    double W;
    string fo_path_parameter; //path to directory for saving trained model
    string fo_path_generation; //path to directory for saving generated sequence
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define and initialize variables
    //dynamic variables and related variables
    //noise
    vector<vector<vector<double> > > eps;
    //prior N(0,1)
    vector<vector<vector<double> > > p_mu;
    vector<vector<vector<double> > > p_sigma;
    //adaptive vector (posterior)
    vector<vector<vector<double> > > a_mu;
    vector<vector<vector<double> > > q_mu;
    vector<vector<vector<double> > > grad_a_mu;
    vector<vector<vector<double> > > adam_m_a_mu;
    vector<vector<vector<double> > > adam_v_a_mu;
    vector<vector<vector<double> > > a_sigma;
    vector<vector<vector<double> > > q_sigma;
    vector<vector<vector<double> > > grad_a_sigma;
    vector<vector<vector<double> > > adam_m_a_sigma;
    vector<vector<vector<double> > > adam_v_a_sigma;
    //latent state
    vector<vector<vector<double> > > z;
    //KLD
    vector<vector<vector<double> > > kld;
    vector<vector<vector<double> > > wkld;
    
    PVRNNTopLayerLatent(int seq_num, int initial_length, int z_num, double w, string& fo_path_param, string& fo_path_gen, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        z_size = z_num;
        W = w;
        if(fi_path_param=="learning"){
            fo_path_parameter = fo_path_param;
            mkdir(fo_path_param.c_str(), 0777);
        }
        fo_path_generation = fo_path_gen;
        mkdir(fo_path_gen.c_str(), 0777);
        fi_path_parameter = fi_path_param;
        //dynamic variables and related variables
        //noise
        eps.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        //prior N(0,1)
        p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 1)));
        //adaptive vector (posterior)
        a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        z.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        kld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        wkld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        /*
        if(fi_path_param=="learning"){
        }else{ //set trained model (example)
            vector<vector<vector<double> > > dummy_mu(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
            vector<vector<vector<double> > > dummy_sigma(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
            for(int s=0;s<seq_num;++s){
                stringstream str_a_mu, str_a_sigma;
                str_a_mu << "./learning_model/" << fi_path_param << "/a_mu_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                str_a_sigma << "./learning_model/" << fi_path_param << "/a_sigma_" << setw(7) << setfill('0') << s << "_" << trained_model_index << ".txt";
                set_weight(dummy_mu[s], str_a_mu.str());
                set_weight(dummy_sigma[s], str_a_sigma.str());
            }
            set_initial_adaptive_vector(a_mu, dummy_mu);
            set_initial_adaptive_vector(a_sigma, dummy_sigma);
        }
        */
    }
    void set_eps(){//used for ensuring reproducibility of simulation
        normal_distribution<> dist(0, 1);
        for(int s=0;s<seq_size;++s){
            for(int t=0;t<init_length;++t){
                for(int i=0;i<z_size;++i){
                    eps[s][t][i] = dist(engine);
                }
            }
        }
    }
    void forward(int t, int s, string mode="posterior"){
        //compute latent states
        for(int i=0;i<z_size;++i){
            if(t>0){
                a_mu[s][t][i] = a_mu[s][t-1][i];
                a_sigma[s][t][i] = a_sigma[s][t-1][i];
            }
            q_mu[s][t][i] = tanh(a_mu[s][t][i]);
            q_sigma[s][t][i] = exp(a_sigma[s][t][i])+ETA;
            if(mode=="posterior"){
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }else if(mode=="prior"){
                    z[s][t][i] = p_mu[s][t][i]+eps[s][t][i]*p_sigma[s][t][i];
            }else{//posterior generation
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }
        }
    }
    void backward(vector<vector<double> >& weight_lz, vector<vector<vector<double> > >& grad_internal_state_lower, vector<double>& tau_lower, int length, int t, int s){ //weight_ld: connections to lower neurons, grad_internal_state_lower: gradient of internal states of lower neurons
        int l_size = int(weight_lz.size());
        double store1=0, store2=0, store3=0, store4=0;
        double normalize = 1.0*init_length/(length*z_size);
        for(int i=0;i<z_size;++i){
            if(t==length-1){
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_lz[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*store1;
                grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1;
            }else if(t==0){
                //error from lower neurons
                kld[s][t][i] = (log(p_sigma[s][t][i])-log(q_sigma[s][t][i])+0.5*(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2)-0.5)*normalize;
                wkld[s][t][i] = W*kld[s][t][i];
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_lz[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*(store1+W*(q_mu[s][t][i]-p_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize)+grad_a_mu[s][t+1][i];
                grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+W*(-1.0+pow(q_sigma[s][t][i],2)/pow(p_sigma[s][t][i],2))*normalize+grad_a_sigma[s][t+1][i];
            }else{
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_lz[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*store1+grad_a_mu[s][t+1][i];
                grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+grad_a_sigma[s][t+1][i];
            }
        }
    }
    void update_parameter_radam(double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-adam2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(adam2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/ (rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        int t = 0;
        for(int s=0;s<seq_size;++s){
            for(int i=0;i<z_size;++i){
                adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
                adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
                m_store = adam_m_a_mu[s][t][i]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_a_mu[s][t][i]+ETA));
                    a_mu[s][t][i] = a_mu[s][t][i]-alpha*r_store*m_store*l_store;
                }else{
                    a_mu[s][t][i] = a_mu[s][t][i]-alpha*m_store;
                }
                adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
                adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
                m_store = adam_m_a_sigma[s][t][i]*m_adam1;
                if(rho>4){
                    l_store = sqrt(l_adam2/(adam_v_a_sigma[s][t][i]+ETA));
                    a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*r_store*m_store*l_store;
                }else{
                    a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*m_store;
                }
            }
        }
    }
    void update_parameter_adam(double alpha, double adam1, double adam2, double m_adam1, double v_adam2){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        int t = 0;
        for(int s=0;s<seq_size;++s){
            for(int i=0;i<z_size;++i){
                adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
                adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
                m_store = adam_m_a_mu[s][t][i]*m_adam1;
                v_store = adam_v_a_mu[s][t][i]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                a_mu[s][t][i] = a_mu[s][t][i]-delta_store; 
                adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
                adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
                m_store = adam_m_a_sigma[s][t][i]*m_adam1;
                v_store = adam_v_a_sigma[s][t][i]*v_adam2;
                delta_store = alpha*m_store/(sqrt(v_store)+ETA);
                a_sigma[s][t][i] = a_sigma[s][t][i]-delta_store;
            }
        }
    }
    void save_sequence(int length, int s, int epoch){
        save_generated_sequence(fo_path_generation, "p_mu", p_mu, length, s, epoch);
        save_generated_sequence(fo_path_generation, "p_sigma", p_sigma, length, s, epoch);
        if(fi_path_parameter=="learning"){
            save_generated_sequence(fo_path_parameter, "a_mu", a_mu, length, s, epoch);
            save_generated_sequence(fo_path_parameter, "a_sigma", a_sigma, length, s, epoch);
        }else{
            save_generated_sequence(fo_path_generation, "a_mu", a_mu, length, s, epoch);
            save_generated_sequence(fo_path_generation, "a_sigma", a_sigma, length, s, epoch);
        }
        save_generated_sequence(fo_path_generation, "z", z, length, s, epoch);
        save_generated_sequence(fo_path_generation, "wkld", wkld, length, s, epoch);
    }
};


//error regression in simulation
class ER_Output{
private:
    int seq_size;
    int init_length;
    int out_size;
    int hd_size; //size of inputs from higher neurons
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    //symaptic weights and related variables
    vector<vector<double> > weight_ohd;
    //time constant
    vector<double> tau;
    //dynamic variables
    vector<vector<vector<double> > > internal_state_output;
    vector<vector<vector<double> > > output;
    vector<vector<vector<double> > > grad_internal_state_output;
    vector<vector<vector<double> > > prediction_error;
    //initialize variables
    ER_Output(int seq_num, int initial_length, int out_num, int hd_num, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        out_size = out_num;
        hd_size = hd_num;
        fi_path_parameter = fi_path_param;
        weight_ohd.assign(out_num, vector<double>(hd_num, 0));
        tau.assign(out_num, 1); //used in backprop at higher level
        internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        grad_internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        prediction_error.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        //initialization
        stringstream str;
        str << "./learning_model/" << fi_path_param << "/weight_ohd_" << trained_model_index << ".txt";
        set_weight(weight_ohd, str.str());
    }
    //error regression
    void er_increment_timestep(int s){
        internal_state_output[s].emplace_back(vector<double>(out_size,0));
        output[s].emplace_back(vector<double>(out_size,0));
        grad_internal_state_output[s].emplace_back(vector<double>(out_size,0));
        prediction_error[s].emplace_back(vector<double>(out_size,0));
    }
    void er_forward(vector<vector<vector<double> > >& hd, int t, int s){ 
        //hd: input from higher neurons, t: time step, s: sequence
        double store1=0;
        for(int i=0;i<out_size;++i){
            store1=0;
            for(int j=0;j<hd_size;++j){
                store1 += weight_ohd[i][j] * hd[s][t][j];
            }
            internal_state_output[s][t][i] = store1;
            output[s][t][i] = tanh(store1);
        }
    }
    void er_backward(vector<vector<vector<double> > >& target, int t, int s){
        //compute gradient of internal state
        double normalize = 1.0/out_size;
        for(int i=0;i<out_size;++i){
            prediction_error[s][t][i] = 0.5*pow(target[s][t][i]-output[s][t][i],2)*normalize;
            grad_internal_state_output[s][t][i] = -(1-pow(output[s][t][i],2))*(target[s][t][i]-output[s][t][i])*normalize;
        }
    }
    void er_save_sequence(string& path_save_generation, vector<vector<vector<double> > >& target, int length, int s, int epoch){
        save_generated_sequence(path_save_generation, "target", target, length, s, epoch);
        save_generated_sequence(path_save_generation, "output", output, length, s, epoch);
        save_generated_sequence(path_save_generation, "pe", prediction_error, length, s, epoch);
    }
};

class ER_Stochastic_Output{//error regression
private:
    int seq_size;
    int init_length;
    int mean_size; //only mean or sigma
    int out_size; //mean+sigma
    int hd_size; //size of inputs from higher neurons
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    //symaptic weights and related variables
    vector<vector<double> > weight_ohd;
    //time constant
    vector<double> tau;
    //dynamic variables
    vector<vector<vector<double> > > internal_state_output;
    vector<vector<vector<double> > > output;
    vector<vector<vector<double> > > grad_internal_state_output;
    vector<vector<vector<double> > > prediction_error; //negative log-likelihood
    //initialize variables
    ER_Stochastic_Output(int seq_num, int initial_length, int out_num, int hd_num, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        mean_size = out_num;
        out_size = 2*out_num;
        hd_size = hd_num;
        fi_path_parameter = fi_path_param;
        //mean:output[0~mean_size-1], sigma:output[mean_size~out_size-1]
        weight_ohd.assign(2*out_num, vector<double>(hd_num, 0));
        tau.assign(2*out_num, 1); //used in backprop at higher level
        internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(2*out_num, 0)));
        output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(2*out_num, 0)));
        grad_internal_state_output.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(2*out_num, 0)));
        prediction_error.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(out_num, 0)));
        //initialization
        stringstream str;
        str << "./learning_model/" << fi_path_param << "/weight_ohd_" << trained_model_index << ".txt";
        set_weight(weight_ohd, str.str());
    }
    
    //error regression
    void er_increment_timestep(int s){
        internal_state_output[s].emplace_back(vector<double>(out_size,0));
        output[s].emplace_back(vector<double>(out_size,0));
        grad_internal_state_output[s].emplace_back(vector<double>(out_size,0));
        prediction_error[s].emplace_back(vector<double>(mean_size,0));
    }
    void er_forward(vector<vector<vector<double> > >& hd, int t, int s){ 
        //hd: input from higher neurons, t: time step, s: sequence
        double store1=0, store2=0; 
        for(int i=0;i<mean_size;++i){
            store1=0, store2=0;
            for(int j=0;j<hd_size;++j){
                store1 += weight_ohd[i][j] * hd[s][t][j];
                store2 += weight_ohd[i+mean_size][j] * hd[s][t][j];
            }
            internal_state_output[s][t][i] = store1;
            internal_state_output[s][t][i+mean_size] = store2;
            output[s][t][i] = tanh(store1);
            output[s][t][i+mean_size] = exp(store2)+ETA;
        }
    }
    void er_backward(vector<vector<vector<double> > >& target, int t, int s, int window_size){
        //compute gradient of internal state
        double normalize = 1.0/(mean_size*window_size);// Please note normalization method that is different from Output class
        for(int i=0;i<mean_size;++i){
            //negative log-likelihood
            prediction_error[s][t][i] = 0.5*(pow((target[s][t][i]-output[s][t][i])/output[s][t][i+mean_size],2)+log(2*PI)+2.0*log(output[s][t][i+mean_size]))*normalize;
            grad_internal_state_output[s][t][i] = -(1-pow(output[s][t][i],2))*(target[s][t][i]-output[s][t][i])/pow(output[s][t][i+mean_size],2)*normalize;
            grad_internal_state_output[s][t][i+mean_size] = (1.0-pow((target[s][t][i]-output[s][t][i])/output[s][t][i+mean_size],2))*normalize;
        }
    }
    void er_save_sequence(string& path_save_generation, vector<vector<vector<double> > >& target, int length, int s, int epoch){
        save_generated_sequence(path_save_generation, "target", target, length, s, epoch);
        save_generated_sequence(path_save_generation, "output", output, length, s, epoch);
        save_generated_sequence(path_save_generation, "pe", prediction_error, length, s, epoch);
    }
};

class ER_PVRNNLayer{
private:
    int seq_size;
    int init_length;
    int d_size;
    int hd_size; //size of inputs from higher deterministic neurons
    int z_size;
    double W;
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    vector<vector<double> > weight_pmd;
    //connections to generate prior sigma
    vector<vector<double> > weight_psd;
    //connections from latent states
    vector<vector<double> > weight_dz;
    //top-down connections
    vector<vector<double> > weight_dhd;
    //recurrent connections
    vector<vector<double> > weight_dd;
    //bias
    vector<double> bias;
    //time constant
    vector<double> tau;
    //dynamic variables and related variables
    //noise
    vector<vector<vector<double> > > eps;
    //prior
    vector<vector<vector<double> > > internal_state_p_mu;
    vector<vector<vector<double> > > p_mu;
    vector<vector<vector<double> > > grad_internal_state_p_mu;
    vector<vector<vector<double> > > internal_state_p_sigma;
    vector<vector<vector<double> > > p_sigma;
    vector<vector<vector<double> > > grad_internal_state_p_sigma;
    //adaptive vector (posterior)
    vector<vector<vector<double> > > a_mu;
    vector<vector<vector<double> > > q_mu;
    vector<vector<vector<double> > > grad_a_mu;
    vector<vector<vector<double> > > adam_m_a_mu;
    vector<vector<vector<double> > > adam_v_a_mu;
    vector<vector<vector<double> > > a_sigma;
    vector<vector<vector<double> > > q_sigma;
    vector<vector<vector<double> > > grad_a_sigma;
    vector<vector<vector<double> > > adam_m_a_sigma;
    vector<vector<vector<double> > > adam_v_a_sigma;
    //latent state
    vector<vector<vector<double> > > z;
    //deterministic state
    vector<vector<vector<double> > > internal_state_d;
    vector<vector<vector<double> > > d;
    vector<vector<vector<double> > > previous_internal_state_d; //if not 0, initial priors can be updated through optimization of synaptic weights 
    vector<vector<vector<double> > > previous_d;
    vector<vector<vector<double> > > grad_internal_state_d;
    //KLD
    vector<vector<vector<double> > > kld;
    vector<vector<vector<double> > > wkld;
    //initialize variables
    ER_PVRNNLayer(int seq_num, int initial_length, int d_num, int hd_num, int z_num, double w, double time_constant, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        d_size = d_num;
        hd_size = hd_num;
        z_size = z_num;
        W = w;
        fi_path_parameter = fi_path_param;
        weight_pmd.assign(z_num, vector<double>(d_num, 0));
        weight_psd.assign(z_num, vector<double>(d_num, 0));
        weight_dz.assign(d_num, vector<double>(z_num, 0));
        weight_dhd.assign(d_num, vector<double>(hd_num, 0));
        weight_dd.assign(d_num, vector<double>(d_num, 0));
        bias.assign(d_num, 0);
        tau.assign(d_num, time_constant);
        for(int i=0;i<d_num;++i){
            if(i>=0.5*d_num){
                tau[i]=2*time_constant;
                }
        }
        eps.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        z.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        previous_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 1))); //if not 0, initial priors can be updated through optimization of synaptic weights 
        previous_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        grad_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        kld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        wkld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        //initialization  //set trained model (example)
        stringstream str_pmd, str_psd, str_dz, str_dhd, str_dd, str_bias;
        str_pmd << "./learning_model/" << fi_path_param << "/weight_pmd_" << trained_model_index << ".txt";
        str_psd << "./learning_model/" << fi_path_param << "/weight_psd_" << trained_model_index << ".txt";
        str_dz << "./learning_model/" << fi_path_param << "/weight_dz_" << trained_model_index << ".txt";
        str_dhd << "./learning_model/" << fi_path_param << "/weight_dhd_" << trained_model_index << ".txt";
        str_dd << "./learning_model/" << fi_path_param << "/weight_dd_" << trained_model_index << ".txt";
        str_bias << "./learning_model/" << fi_path_param << "/bias_" << trained_model_index << ".txt";
        set_weight(weight_pmd, str_pmd.str());
        set_weight(weight_psd, str_psd.str());
        set_weight(weight_dz, str_dz.str());
        set_weight(weight_dhd, str_dhd.str());
        set_weight(weight_dd, str_dd.str());
        set_bias(bias, str_bias.str());
    }

    //error regression
    void er_increment_timestep(int s){
        eps[s].emplace_back(vector<double>(z_size,0));
        internal_state_p_mu[s].emplace_back(vector<double>(z_size,0));
        p_mu[s].emplace_back(vector<double>(z_size,0));
        grad_internal_state_p_mu[s].emplace_back(vector<double>(z_size,0));
        internal_state_p_sigma[s].emplace_back(vector<double>(z_size,0));
        p_sigma[s].emplace_back(vector<double>(z_size,0));
        grad_internal_state_p_sigma[s].emplace_back(vector<double>(z_size,0));
        a_mu[s].emplace_back(vector<double>(z_size,0));
        q_mu[s].emplace_back(vector<double>(z_size,0));
        grad_a_mu[s].emplace_back(vector<double>(z_size,0));
        adam_m_a_mu[s].emplace_back(vector<double>(z_size,0));
        adam_v_a_mu[s].emplace_back(vector<double>(z_size,0));
        a_sigma[s].emplace_back(vector<double>(z_size,0));
        q_sigma[s].emplace_back(vector<double>(z_size,0));
        grad_a_sigma[s].emplace_back(vector<double>(z_size,0));
        adam_m_a_sigma[s].emplace_back(vector<double>(z_size,0));
        adam_v_a_sigma[s].emplace_back(vector<double>(z_size,0));
        z[s].emplace_back(vector<double>(z_size,0));
        internal_state_d[s].emplace_back(vector<double>(d_size,0));
        d[s].emplace_back(vector<double>(d_size,0));
        previous_internal_state_d[s].emplace_back(vector<double>(d_size,0));
        previous_d[s].emplace_back(vector<double>(d_size,0));
        grad_internal_state_d[s].emplace_back(vector<double>(d_size,0));
        kld[s].emplace_back(vector<double>(z_size,0));
        wkld[s].emplace_back(vector<double>(z_size,0));
    }
    void er_forward(vector<vector<vector<double> > >& hd, int t, int s, int epoch, int initial_regression, int last_window_step, string mode="posterior"){
        double store1=0, store2=0, store3=0;
        normal_distribution<> dist(0, 1);
        //set previous deterministic states
        for(int i=0;i<d_size;++i){
            if(t != 0) previous_internal_state_d[s][t][i] = internal_state_d[s][t-1][i];
            previous_d[s][t][i] = tanh(previous_internal_state_d[s][t][i]);
        }
        //compute latent states
        for(int i=0;i<z_size;++i){
            store1=0.0;
            store2=0.0;
            for(int j=0;j<d_size;++j){
                store1 += weight_pmd[i][j]*previous_d[s][t][j];
                store2 += weight_psd[i][j]*previous_d[s][t][j];
            }
            internal_state_p_mu[s][t][i] = store1;
            internal_state_p_sigma[s][t][i] = store2;
            p_mu[s][t][i] = tanh(store1);
            p_sigma[s][t][i] = exp(store2)+ETA;
            //initialize internal posterior state with initial internal prior state
            if(epoch==0 && initial_regression==1){
                //*if you set initial adaptive vector externally, please comment out the following two lines.
                a_mu[s][t][i] = store1;
                a_sigma[s][t][i] = store2;
            }else if(epoch==0 && last_window_step==1){
                a_mu[s][t][i] = store1;
                a_sigma[s][t][i] = store2;
            }
            q_mu[s][t][i] = tanh(a_mu[s][t][i]);
            q_sigma[s][t][i] = exp(a_sigma[s][t][i])+ETA;
            eps[s][t][i] = dist(engine);
            if(mode=="posterior"){
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }else if(mode=="prior"){
                    z[s][t][i] = p_mu[s][t][i]+eps[s][t][i]*p_sigma[s][t][i];
            }else{//posterior generation
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }
        }
        //compute deterministic states
        for(int i=0;i<d_size;++i){
            //recurrent connections
            store1=0;
            for(int j=0;j<d_size;++j){
                store1 += weight_dd[i][j]*previous_d[s][t][j];
            }
            //top-down connections
            store2=0;
            for(int j=0;j<hd_size;++j){
                store2 += weight_dhd[i][j]*hd[s][t][j];
            }
            //connections from latent states
            store3=0;
            for(int j=0;j<z_size;++j){
                store3 += weight_dz[i][j]*z[s][t][j];
            }
            internal_state_d[s][t][i] = (1.0-1.0/tau[i])*previous_internal_state_d[s][t][i]+1.0/tau[i]*(store1+store2+store3+bias[i]);
            d[s][t][i] = tanh(internal_state_d[s][t][i]);
        }
    }
    void er_backward(vector<vector<double> >& weight_ld, vector<vector<vector<double> > >& grad_internal_state_lower, vector<double>& tau_lower, int last_window_step, int t, int s){ //weight_ld: connections to lower neurons, grad_internal_state_lower: gradient of internal states of lower neurons
        int l_size = int(weight_ld.size());
        double store1=0, store2=0, store3=0, store4=0;
        double normalize = 1.0/z_size;
        for(int i=0;i<d_size;++i){
            //compute gradient of internal state of deterministic neuron
            if(last_window_step==1){
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*store1;
            }else{
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                //error from next deterministic states
                store2=0;
                for(int k=0;k<d_size;++k){
                    store2 += 1.0/tau[k]*weight_dd[k][i]*grad_internal_state_d[s][t+1][k];
                }
                //error from next prior states
                store3=0, store4=0;
                for(int k=0;k<z_size;++k){
                    store3 += weight_pmd[k][i]*grad_internal_state_p_mu[s][t+1][k];
                    store4 += weight_psd[k][i]*grad_internal_state_p_sigma[s][t+1][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*(store1+store2+store3+store4)+(1.0-1.0/tau[i])*grad_internal_state_d[s][t+1][i];
            }
        }
        //compute gradient of parameters related with prior and posterior
        for(int i=0;i<z_size;++i){
            kld[s][t][i] = (log(p_sigma[s][t][i])-log(q_sigma[s][t][i])+0.5*(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2)-0.5)*normalize;
            wkld[s][t][i] = W*kld[s][t][i];
            store1=0;
            for(int k=0;k<d_size;++k){
                store1 += 1.0/tau[k]*weight_dz[k][i]*grad_internal_state_d[s][t][k]; //gradient of z
            }
            grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*(store1+W*(q_mu[s][t][i]-p_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize);
            grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+ W*(-1.0+pow(q_sigma[s][t][i],2)/pow(p_sigma[s][t][i],2))*normalize;
            grad_internal_state_p_mu[s][t][i] = (1.0-pow(p_mu[s][t][i],2))*W*(p_mu[s][t][i]-q_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize;
            grad_internal_state_p_sigma[s][t][i] = W*(1.0-(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2))*normalize;
        }
    }
    void er_update_parameter_radam(int t, int s, int epoch, double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-adam2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(adam2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/(rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        for(int i=0;i<z_size;++i){
            if(epoch==0){
                adam_m_a_mu[s][t][i] = 0.0;
                adam_v_a_mu[s][t][i] = 0.0;
                adam_m_a_sigma[s][t][i] = 0.0;
                adam_v_a_sigma[s][t][i] = 0.0;
            }
            adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
            adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
            m_store = adam_m_a_mu[s][t][i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_a_mu[s][t][i]+ETA));
                a_mu[s][t][i] = a_mu[s][t][i]-alpha*r_store*m_store*l_store;
            }else{
                a_mu[s][t][i] = a_mu[s][t][i]-alpha*m_store;
            }
            adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
            adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
            m_store = adam_m_a_sigma[s][t][i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_a_sigma[s][t][i]+ETA));
                a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*r_store*m_store*l_store;
            }else{
                a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*m_store;
            }                 
        }
    }
    void er_update_parameter_adam(int t, int s, int epoch, double alpha, double adam1, double adam2, double m_adam1, double v_adam2){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        for(int i=0;i<z_size;++i){
            if(epoch==0){
                adam_m_a_mu[s][t][i] = 0.0;
                adam_v_a_mu[s][t][i] = 0.0;
                adam_m_a_sigma[s][t][i] = 0.0;
                adam_v_a_sigma[s][t][i] = 0.0;
            }
            adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
            adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
            m_store = adam_m_a_mu[s][t][i]*m_adam1;
            v_store = adam_v_a_mu[s][t][i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            a_mu[s][t][i] = a_mu[s][t][i]-delta_store; 
            adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
            adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
            m_store = adam_m_a_sigma[s][t][i]*m_adam1;
            v_store = adam_v_a_sigma[s][t][i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            a_sigma[s][t][i] = a_sigma[s][t][i]-delta_store;                    
        }
    }
    void er_save_sequence(string& path_save_generation, int length, int s, int epoch){
        save_generated_sequence(path_save_generation, "in_p_mu", internal_state_p_mu, length, s, epoch);
        save_generated_sequence(path_save_generation, "in_p_sigma", internal_state_p_sigma, length, s, epoch);
        save_generated_sequence(path_save_generation, "a_mu", a_mu, length, s, epoch);
        save_generated_sequence(path_save_generation, "a_sigma", a_sigma, length, s, epoch);
        save_generated_sequence(path_save_generation, "z", z, length, s, epoch);
        save_generated_sequence(path_save_generation, "in_d", internal_state_d, length, s, epoch);
        save_generated_sequence(path_save_generation, "wkld", wkld, length, s, epoch);
    }
    
};

class ER_PVRNNTopLayer{
private:
    int seq_size;
    int init_length;
    int d_size;
    int z_size;
    double W;
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define variables
    vector<vector<double> > weight_pmd;
    //connections to generate prior sigma
    vector<vector<double> > weight_psd;
    //connections from latent states
    vector<vector<double> > weight_dz;
    //recurrent connections
    vector<vector<double> > weight_dd;
    //bias
    vector<double> bias;
    //time constant
    vector<double> tau;
    //dynamic variables and related variables
    //noise
    vector<vector<vector<double> > > eps;
    //prior
    vector<vector<vector<double> > > internal_state_p_mu;
    vector<vector<vector<double> > > p_mu;
    vector<vector<vector<double> > > grad_internal_state_p_mu;
    vector<vector<vector<double> > > internal_state_p_sigma;
    vector<vector<vector<double> > > p_sigma;
    vector<vector<vector<double> > > grad_internal_state_p_sigma;
    //adaptive vector (posterior)
    vector<vector<vector<double> > > a_mu;
    vector<vector<vector<double> > > q_mu;
    vector<vector<vector<double> > > grad_a_mu;
    vector<vector<vector<double> > > adam_m_a_mu;
    vector<vector<vector<double> > > adam_v_a_mu;
    vector<vector<vector<double> > > a_sigma;
    vector<vector<vector<double> > > q_sigma;
    vector<vector<vector<double> > > grad_a_sigma;
    vector<vector<vector<double> > > adam_m_a_sigma;
    vector<vector<vector<double> > > adam_v_a_sigma;
    //latent state
    vector<vector<vector<double> > > z;
    //deterministic state
    vector<vector<vector<double> > > internal_state_d;
    vector<vector<vector<double> > > d;
    vector<vector<vector<double> > > previous_internal_state_d; //if not 0, initial priors can be updated through optimization of synaptic weights 
    vector<vector<vector<double> > > previous_d;
    vector<vector<vector<double> > > grad_internal_state_d;
    //KLD
    vector<vector<vector<double> > > kld;
    vector<vector<vector<double> > > wkld;
    //initialize variables
    ER_PVRNNTopLayer(int seq_num, int initial_length, int d_num, int z_num, double w, double time_constant, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        d_size = d_num;
        z_size = z_num;
        W = w;
        fi_path_parameter = fi_path_param;
        weight_pmd.assign(z_num, vector<double>(d_num, 0));
        weight_psd.assign(z_num, vector<double>(d_num, 0));
        weight_dz.assign(d_num, vector<double>(z_num, 0));
        weight_dd.assign(d_num, vector<double>(d_num, 0));
        bias.assign(d_num, 0);
        tau.assign(d_num, time_constant);
        for(int i=0;i<d_num;++i){
            if(i>=0.5*d_num){
                tau[i]=2*time_constant;
                }
        }
        eps.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_internal_state_p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        z.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        previous_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 1))); //if not 0, initial priors can be updated through optimization of synaptic weights 
        previous_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        grad_internal_state_d.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(d_num, 0)));
        kld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        wkld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        //initialization  //set trained model (example)
        stringstream str_pmd, str_psd, str_dz, str_dd, str_bias;
        str_pmd << "./learning_model/" << fi_path_param << "/weight_pmd_" << trained_model_index << ".txt";
        str_psd << "./learning_model/" << fi_path_param << "/weight_psd_" << trained_model_index << ".txt";
        str_dz << "./learning_model/" << fi_path_param << "/weight_dz_" << trained_model_index << ".txt";
        str_dd << "./learning_model/" << fi_path_param << "/weight_dd_" << trained_model_index << ".txt";
        str_bias << "./learning_model/" << fi_path_param << "/bias_" << trained_model_index << ".txt";
        set_weight(weight_pmd, str_pmd.str());
        set_weight(weight_psd, str_psd.str());
        set_weight(weight_dz, str_dz.str());
        set_weight(weight_dd, str_dd.str());
        set_bias(bias, str_bias.str());
    }
    //error regression
    void er_increment_timestep(int s){
        eps[s].emplace_back(vector<double>(z_size,0));
        internal_state_p_mu[s].emplace_back(vector<double>(z_size,0));
        p_mu[s].emplace_back(vector<double>(z_size,0));
        grad_internal_state_p_mu[s].emplace_back(vector<double>(z_size,0));
        internal_state_p_sigma[s].emplace_back(vector<double>(z_size,0));
        p_sigma[s].emplace_back(vector<double>(z_size,0));
        grad_internal_state_p_sigma[s].emplace_back(vector<double>(z_size,0));
        a_mu[s].emplace_back(vector<double>(z_size,0));
        q_mu[s].emplace_back(vector<double>(z_size,0));
        grad_a_mu[s].emplace_back(vector<double>(z_size,0));
        adam_m_a_mu[s].emplace_back(vector<double>(z_size,0));
        adam_v_a_mu[s].emplace_back(vector<double>(z_size,0));
        a_sigma[s].emplace_back(vector<double>(z_size,0));
        q_sigma[s].emplace_back(vector<double>(z_size,0));
        grad_a_sigma[s].emplace_back(vector<double>(z_size,0));
        adam_m_a_sigma[s].emplace_back(vector<double>(z_size,0));
        adam_v_a_sigma[s].emplace_back(vector<double>(z_size,0));
        z[s].emplace_back(vector<double>(z_size,0));
        internal_state_d[s].emplace_back(vector<double>(d_size,0));
        d[s].emplace_back(vector<double>(d_size,0));
        previous_internal_state_d[s].emplace_back(vector<double>(d_size,0));
        previous_d[s].emplace_back(vector<double>(d_size,0));
        grad_internal_state_d[s].emplace_back(vector<double>(d_size,0));
        kld[s].emplace_back(vector<double>(z_size,0));
        wkld[s].emplace_back(vector<double>(z_size,0));
    }
    void er_forward(int t, int s, int epoch, int initial_regression, int last_window_step, string mode="posterior"){
        double store1=0, store2=0, store3=0;
        normal_distribution<> dist(0, 1);
        //set previous deterministic states
        for(int i=0;i<d_size;++i){
            if(t != 0) previous_internal_state_d[s][t][i] = internal_state_d[s][t-1][i];
            previous_d[s][t][i] = tanh(previous_internal_state_d[s][t][i]);
        }
        //compute latent states
        for(int i=0;i<z_size;++i){
            store1=0.0;
            store2=0.0;
            for(int j=0;j<d_size;++j){
                store1 += weight_pmd[i][j]*previous_d[s][t][j];
                store2 += weight_psd[i][j]*previous_d[s][t][j];
            }
            internal_state_p_mu[s][t][i] = store1;
            internal_state_p_sigma[s][t][i] = store2;
            p_mu[s][t][i] = tanh(store1);
            p_sigma[s][t][i] = exp(store2)+ETA;
            //initialize internal posterior state with initial internal prior state
            if(epoch==0 && initial_regression==1){
                //*if you set initial adaptive vector externally, please comment out the following two lines.
                a_mu[s][t][i] = store1;
                a_sigma[s][t][i] = store2;
            }else if(epoch==0 && last_window_step==1){
                a_mu[s][t][i] = store1;
                a_sigma[s][t][i] = store2;
            }
            q_mu[s][t][i] = tanh(a_mu[s][t][i]);
            q_sigma[s][t][i] = exp(a_sigma[s][t][i])+ETA;
            eps[s][t][i] = dist(engine);
            if(mode=="posterior"){
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }else if(mode=="prior"){
                    z[s][t][i] = p_mu[s][t][i]+eps[s][t][i]*p_sigma[s][t][i];
            }else{//posterior generation
                    z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
            }
        }
        //compute deterministic states
        for(int i=0;i<d_size;++i){
            //recurrent connections
            store1=0;
            for(int j=0;j<d_size;++j){
                store1 += weight_dd[i][j]*previous_d[s][t][j];
            }
            //connections from latent states
            store3=0;
            for(int j=0;j<z_size;++j){
                store3 += weight_dz[i][j]*z[s][t][j];
            }
            internal_state_d[s][t][i] = (1.0-1.0/tau[i])*previous_internal_state_d[s][t][i]+1.0/tau[i]*(store1+store3+bias[i]);
            d[s][t][i] = tanh(internal_state_d[s][t][i]);
        }
    }
    void er_backward(vector<vector<double> >& weight_ld, vector<vector<vector<double> > >& grad_internal_state_lower, vector<double>& tau_lower, int last_window_step, int t, int s){ //weight_ld: connections to lower neurons, grad_internal_state_lower: gradient of internal states of lower neurons
        int l_size = int(weight_ld.size());
        double store1=0, store2=0, store3=0, store4=0;
        double normalize = 1.0/z_size;
        for(int i=0;i<d_size;++i){
            //compute gradient of internal state of deterministic neuron
            if(last_window_step==1){
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*store1;
            }else{
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_ld[k][i]*grad_internal_state_lower[s][t][k];
                }
                //error from next deterministic states
                store2=0;
                for(int k=0;k<d_size;++k){
                    store2 += 1.0/tau[k]*weight_dd[k][i]*grad_internal_state_d[s][t+1][k];
                }
                //error from next prior states
                store3=0, store4=0;
                for(int k=0;k<z_size;++k){
                    store3 += weight_pmd[k][i]*grad_internal_state_p_mu[s][t+1][k];
                    store4 += weight_psd[k][i]*grad_internal_state_p_sigma[s][t+1][k];
                }
                grad_internal_state_d[s][t][i] = (1.0-pow(d[s][t][i],2))*(store1+store2+store3+store4)+(1.0-1.0/tau[i])*grad_internal_state_d[s][t+1][i];
            }
        }
        //compute gradient of parameters related with prior and posterior
        for(int i=0;i<z_size;++i){
            kld[s][t][i] = (log(p_sigma[s][t][i])-log(q_sigma[s][t][i])+0.5*(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2)-0.5)*normalize;
            wkld[s][t][i] = W*kld[s][t][i];
            store1=0;
            for(int k=0;k<d_size;++k){
                store1 += 1.0/tau[k]*weight_dz[k][i]*grad_internal_state_d[s][t][k]; //gradient of z
            }
            grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*(store1+W*(q_mu[s][t][i]-p_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize);
            grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+ W*(-1.0+pow(q_sigma[s][t][i],2)/pow(p_sigma[s][t][i],2))*normalize;
            grad_internal_state_p_mu[s][t][i] = (1.0-pow(p_mu[s][t][i],2))*W*(p_mu[s][t][i]-q_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize;
            grad_internal_state_p_sigma[s][t][i] = W*(1.0-(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2))*normalize;
        }
    }
    void er_update_parameter_radam(int t, int s, int epoch, double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-adam2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(adam2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/(rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        for(int i=0;i<z_size;++i){
            if(epoch==0){
                adam_m_a_mu[s][t][i] = 0.0;
                adam_v_a_mu[s][t][i] = 0.0;
                adam_m_a_sigma[s][t][i] = 0.0;
                adam_v_a_sigma[s][t][i] = 0.0;
            }
            adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
            adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
            m_store = adam_m_a_mu[s][t][i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_a_mu[s][t][i]+ETA));
                a_mu[s][t][i] = a_mu[s][t][i]-alpha*r_store*m_store*l_store;
            }else{
                a_mu[s][t][i] = a_mu[s][t][i]-alpha*m_store;
            }
            adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
            adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
            m_store = adam_m_a_sigma[s][t][i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_a_sigma[s][t][i]+ETA));
                a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*r_store*m_store*l_store;
            }else{
                a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*m_store;
            }                 
         }
    }
    void er_update_parameter_adam(int t, int s, int epoch, double alpha, double adam1, double adam2, double m_adam1, double v_adam2){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        for(int i=0;i<z_size;++i){
            if(epoch==0){
                adam_m_a_mu[s][t][i] = 0.0;
                adam_v_a_mu[s][t][i] = 0.0;
                adam_m_a_sigma[s][t][i] = 0.0;
                adam_v_a_sigma[s][t][i] = 0.0;
            }
            adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
            adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
            m_store = adam_m_a_mu[s][t][i]*m_adam1;
            v_store = adam_v_a_mu[s][t][i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            a_mu[s][t][i] = a_mu[s][t][i]-delta_store; 
            adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
            adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
            m_store = adam_m_a_sigma[s][t][i]*m_adam1;
            v_store = adam_v_a_sigma[s][t][i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            a_sigma[s][t][i] = a_sigma[s][t][i]-delta_store;                    
        }
    }
    void er_save_sequence(string& path_save_generation, int length, int s, int epoch){
        save_generated_sequence(path_save_generation, "in_p_mu", internal_state_p_mu, length, s, epoch);
        save_generated_sequence(path_save_generation, "in_p_sigma", internal_state_p_sigma, length, s, epoch);
        save_generated_sequence(path_save_generation, "a_mu", a_mu, length, s, epoch);
        save_generated_sequence(path_save_generation, "a_sigma", a_sigma, length, s, epoch);
        save_generated_sequence(path_save_generation, "z", z, length, s, epoch);
        save_generated_sequence(path_save_generation, "in_d", internal_state_d, length, s, epoch);
        save_generated_sequence(path_save_generation, "wkld", wkld, length, s, epoch);
    }
    
};

class ER_PVRNNTopLayerLatent{
private:
    int seq_size;
    int init_length;
    int z_size;
    double W;
    string fi_path_parameter; //path to directory for reading trained model
public:
    //define and initialize variables
    //dynamic variables and related variables
    //noise
    vector<vector<vector<double> > > eps;
    //prior N(0,1)
    vector<vector<vector<double> > > p_mu;
    vector<vector<vector<double> > > p_sigma;
    //adaptive vector (posterior)
    vector<vector<vector<double> > > a_mu;
    vector<vector<vector<double> > > q_mu;
    vector<vector<vector<double> > > grad_a_mu;
    vector<vector<vector<double> > > adam_m_a_mu;
    vector<vector<vector<double> > > adam_v_a_mu;
    vector<vector<vector<double> > > a_sigma;
    vector<vector<vector<double> > > q_sigma;
    vector<vector<vector<double> > > grad_a_sigma;
    vector<vector<vector<double> > > adam_m_a_sigma;
    vector<vector<vector<double> > > adam_v_a_sigma;
    //latent state
    vector<vector<vector<double> > > z;
    //KLD
    vector<vector<vector<double> > > kld;
    vector<vector<vector<double> > > wkld;
    
    ER_PVRNNTopLayerLatent(int seq_num, int initial_length, int z_num, double w, string fi_path_param){
        seq_size = seq_num;
        init_length = initial_length;
        z_size = z_num;
        W = w;
        fi_path_parameter = fi_path_param;
        //dynamic variables and related variables
        //noise
        eps.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        //prior N(0,1)
        p_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        p_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 1)));
        //adaptive vector (posterior)
        a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_mu.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        q_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        grad_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_m_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        adam_v_a_sigma.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        z.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        kld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
        wkld.assign(seq_num, vector<vector<double> >(initial_length, vector<double>(z_num, 0)));
    }
   
    //error regression
    void er_increment_timestep(int s){
        eps[s].emplace_back(vector<double>(z_size,0));
        p_mu[s].emplace_back(vector<double>(z_size,0));
        p_sigma[s].emplace_back(vector<double>(z_size,0));
        a_mu[s].emplace_back(vector<double>(z_size,0));
        q_mu[s].emplace_back(vector<double>(z_size,0));
        grad_a_mu[s].emplace_back(vector<double>(z_size,0));
        adam_m_a_mu[s].emplace_back(vector<double>(z_size,0));
        adam_v_a_mu[s].emplace_back(vector<double>(z_size,0));
        a_sigma[s].emplace_back(vector<double>(z_size,0));
        q_sigma[s].emplace_back(vector<double>(z_size,0));
        grad_a_sigma[s].emplace_back(vector<double>(z_size,0));
        adam_m_a_sigma[s].emplace_back(vector<double>(z_size,0));
        adam_v_a_sigma[s].emplace_back(vector<double>(z_size,0));
        z[s].emplace_back(vector<double>(z_size,0));
        kld[s].emplace_back(vector<double>(z_size,0));
        wkld[s].emplace_back(vector<double>(z_size,0));
    }
    void er_forward(int t, int s, int epoch, int first_window_step, string mode="posterior"){
        normal_distribution<> dist(0, 1);
        //compute latent states
        for(int i=0;i<z_size;++i){
            /*
            if(epoch==0 && t==0){//initialize by unit Gaussian prior N(0,1), or initial prior
                //if you set initial adaptive vector externally, please comment out the following two lines.
                a_mu[s][t][i] = 0.0;
                a_sigma[s][t][i] = 0.0;
            }else if(first_window_step!=1){
                a_mu[s][t][i] = a_mu[s][t-1][i];
                a_sigma[s][t][i] = a_sigma[s][t-1][i];
            }
            */
            if(first_window_step!=1){
                a_mu[s][t][i] = a_mu[s][t-1][i];
                a_sigma[s][t][i] = a_sigma[s][t-1][i];
            }
            q_mu[s][t][i] = tanh(a_mu[s][t][i]);
            q_sigma[s][t][i] = exp(a_sigma[s][t][i])+ETA;
            eps[s][t][i] = dist(engine);
            z[s][t][i] = q_mu[s][t][i]+eps[s][t][i]*q_sigma[s][t][i];
        }
    }
    void er_backward(vector<vector<double> >& weight_lz, vector<vector<vector<double> > >& grad_internal_state_lower, vector<double>& tau_lower, int last_window_step, int t, int s){ //weight_ld: connections to lower neurons, grad_internal_state_lower: gradient of internal states of lower neurons
        int l_size = int(weight_lz.size());
        double store1=0, store2=0, store3=0, store4=0;
        double normalize = 1.0/z_size;
        for(int i=0;i<z_size;++i){
            if(t==0 && last_window_step==1){//occur in real-time robot operation
                kld[s][t][i] = (log(p_sigma[s][t][i])-log(q_sigma[s][t][i])+0.5*(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2)-0.5)*normalize;
                wkld[s][t][i] = W*kld[s][t][i];
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_lz[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*(store1+W*(q_mu[s][t][i]-p_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize);
                grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+W*(-1.0+pow(q_sigma[s][t][i],2)/pow(p_sigma[s][t][i],2))*normalize;
            }else if(last_window_step==1){
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_lz[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*store1;
                grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1;
            }else if(t==0){
                kld[s][t][i] = (log(p_sigma[s][t][i])-log(q_sigma[s][t][i])+0.5*(pow(p_mu[s][t][i]-q_mu[s][t][i],2)+pow(q_sigma[s][t][i],2))/pow(p_sigma[s][t][i],2)-0.5)*normalize;
                wkld[s][t][i] = W*kld[s][t][i];
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_lz[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*(store1+W*(q_mu[s][t][i]-p_mu[s][t][i])/pow(p_sigma[s][t][i],2)*normalize)+grad_a_mu[s][t+1][i];
                grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+W*(-1.0+pow(q_sigma[s][t][i],2)/pow(p_sigma[s][t][i],2))*normalize+grad_a_sigma[s][t+1][i];
            }else{
                //error from lower neurons
                store1=0;
                for(int k=0;k<l_size;++k){
                    store1 += 1.0/tau_lower[k]*weight_lz[k][i]*grad_internal_state_lower[s][t][k];
                }
                grad_a_mu[s][t][i] = (1.0-pow(q_mu[s][t][i],2))*store1+grad_a_mu[s][t+1][i];
                grad_a_sigma[s][t][i] = q_sigma[s][t][i]*eps[s][t][i]*store1+grad_a_sigma[s][t+1][i];
            }
        }
    }
    void er_update_parameter_radam(int t, int s, int epoch, double alpha, double adam1, double adam2, double rho, double r_store, double m_adam1, double l_adam2){
        double m_store=0, l_store=0;
        /*
        double rho_inf = 2.0/(1.0-adam2)-1.0;
        double rho = rho_inf-2.0*(epoch+1)*pow(adam2,epoch+1)/(1.0-pow(adam2,epoch+1));
        double r_store = sqrt((rho-4.0)*(rho-2.0)*rho_inf/ (rho_inf-4.0)/(rho_inf-2.0)/rho);
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double l_adam2 = 1.0-pow(adam2,epoch+1);
        */
        for(int i=0;i<z_size;++i){
            if(epoch==0){
                adam_m_a_mu[s][t][i] = 0.0;
                adam_v_a_mu[s][t][i] = 0.0;
                adam_m_a_sigma[s][t][i] = 0.0;
                adam_v_a_sigma[s][t][i] = 0.0;
            }
            adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
            adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
            m_store = adam_m_a_mu[s][t][i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_a_mu[s][t][i]+ETA));
                a_mu[s][t][i] = a_mu[s][t][i]-alpha*r_store*m_store*l_store;
            }else{
                a_mu[s][t][i] = a_mu[s][t][i]-alpha*m_store;
            }
            adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
            adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
            m_store = adam_m_a_sigma[s][t][i]*m_adam1;
            if(rho>4){
                l_store = sqrt(l_adam2/(adam_v_a_sigma[s][t][i]+ETA));
                a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*r_store*m_store*l_store;
            }else{
                a_sigma[s][t][i] = a_sigma[s][t][i]-alpha*m_store;
            }
        }
    }
    void er_update_parameter_adam(int t, int s, int epoch, double alpha, double adam1, double adam2, double m_adam1, double v_adam2){
        double m_store=0, v_store=0, delta_store=0;
        /*
        double m_adam1 = 1.0/(1.0-pow(adam1,epoch+1));
        double v_adam2 = 1.0/(1.0-pow(adam2,epoch+1));
        */
        for(int i=0;i<z_size;++i){
            if(epoch==0){
                adam_m_a_mu[s][t][i] = 0.0;
                adam_v_a_mu[s][t][i] = 0.0;
                adam_m_a_sigma[s][t][i] = 0.0;
                adam_v_a_sigma[s][t][i] = 0.0;
            }
            adam_m_a_mu[s][t][i] = adam1*adam_m_a_mu[s][t][i]+(1-adam1)*grad_a_mu[s][t][i];
            adam_v_a_mu[s][t][i] = adam2*adam_v_a_mu[s][t][i]+(1-adam2)*pow(grad_a_mu[s][t][i],2);
            m_store = adam_m_a_mu[s][t][i]*m_adam1;
            v_store = adam_v_a_mu[s][t][i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            a_mu[s][t][i] = a_mu[s][t][i]-delta_store; 
            adam_m_a_sigma[s][t][i] = adam1*adam_m_a_sigma[s][t][i]+(1-adam1)*grad_a_sigma[s][t][i];
            adam_v_a_sigma[s][t][i] = adam2*adam_v_a_sigma[s][t][i]+(1-adam2)*pow(grad_a_sigma[s][t][i],2);
            m_store = adam_m_a_sigma[s][t][i]*m_adam1;
            v_store = adam_v_a_sigma[s][t][i]*v_adam2;
            delta_store = alpha*m_store/(sqrt(v_store)+ETA);
            a_sigma[s][t][i] = a_sigma[s][t][i]-delta_store;
        }
    }
    void er_save_sequence(string& path_save_generation, int length, int s, int epoch){
        save_generated_sequence(path_save_generation, "p_mu", p_mu, length, s, epoch);
        save_generated_sequence(path_save_generation, "p_sigma", p_sigma, length, s, epoch);
        save_generated_sequence(path_save_generation, "a_mu", a_mu, length, s, epoch);
        save_generated_sequence(path_save_generation, "a_sigma", a_sigma, length, s, epoch);
        save_generated_sequence(path_save_generation, "z", z, length, s, epoch);
        save_generated_sequence(path_save_generation, "wkld", wkld, length, s, epoch);
    }    
};
