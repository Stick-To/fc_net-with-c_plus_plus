#ifndef __FC_NET_H__
#define __FC_NET_H__
#include<string>
#include<map>
#include"matrix.h"
#include<iostream>
#include<sstream>
class fc_net
{
private:
    typedef std::map<std::string,matrix<double>> map1str;
    typedef std::map<std::string,map1str> map2str;
public:
    explicit fc_net(const int layer[],const int num):num_layers(num)
    {
        for(int i=1;i!=num_layers;++i)
        {
            std::string strW="W"+num2str(i);
            std::string strb="b"+num2str(i);
            matrix<double> W_temp(layer[i-1],layer[i]);
            W_temp.init_xavier();
            matrix<double> b_temp(1,layer[i]);
            b_temp.init_zeros();
            parameter[strW]=W_temp;
            parameter[strb]=b_temp;
        }
    }
    explicit fc_net()=default;
    map1str get_parameter()
    {
        return parameter;
    }
    int get_num_layers()
    {
        return num_layers;
    }
public:
    double train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,
                    const double reg);
    matrix<int> predict(matrix<double> &dataset);
private:
    map1str parameter;
    int num_layers;
private:
    std::string num2str(const double i)
    {
        std::stringstream ss;
        ss<<i;
        return ss.str();
    }
private:
    map1str softmax(matrix<double> &scores,matrix<int> &labels,const double reg);
    map2str affine_forward(matrix<double> &x,matrix<double> &W,matrix<double> &b);
    map1str affine_backward(matrix<double> &upstream_grad,map1str &cache);
    map2str relu_forward(matrix<double> &lhs);
    matrix<double> relu_backward(matrix<double> &upstream_grad,map1str &cache);
    map2str batchnorm_forward(matrix<double> &x,map1str &para,const std::string mode,const double momentum);
    map1str batchnorm_backward(matrix<double> &upstream_grad,map1str &cache);
    map2str dropout_forward(matrix<double> &x, const double p,const std::string mode);
    matrix<double> dropout_backward(matrix<double> &upstream_grad,map1str &cache);

};
fc_net::map1str fc_net::softmax(matrix<double> &scores,matrix<int> &labels,const double reg)
{
    assert((labels.axisy()==1));
    map1str result;
    matrix<double> correct_scores(labels.axisx(),1);
    for(int i=0;i!=labels.axisx();++i)
    {
        correct_scores[i][0]=scores[i][labels[i][0]];
    }

    matrix<double> exp_=(scores-correct_scores).exp();
    matrix<double> exp_sum=exp_.sum(0);

    matrix<double> log_=exp_sum.log();
    matrix<double> loss=matrix<double>(1,1);

    loss[0][0]=log_.sum()/scores.axisx();

    for(int i=1;i!=num_layers;++i)
    {
        loss[0][0]+=reg*(parameter["W"+num2str(i)].square().sum());
    }
    result["loss"]=loss;
    matrix<double> dscores=exp_/exp_sum;
    for(int i=0;i!=labels.axisx();++i)
    {
            dscores[i][labels[i][0]]-=1;
    }
    dscores/=labels.axisx();
    result["dscores"]=dscores;

    for(int i=1;i!=num_layers;++i)
    {
        result["dW"+num2str(i)]=2*reg*parameter["W"+num2str(i)];
    }
    return result;
}


fc_net::map2str fc_net::affine_forward(matrix<double> &x,matrix<double> &W,matrix<double> &b)
{
    map2str result;
    matrix<double> affine=x.dot(W)+b;
    map1str obj;
    map1str cache;
    obj["affine"]=affine;
    cache["x"]=x;
    cache["W"]=W;
    cache["b"]=b;
    result["affine"]=obj;
    result["cache"]=cache;

    return result;
}
fc_net::map2str fc_net::relu_forward(matrix<double> &lhs)
{
    map2str result;
    map1str obj;
    map1str cache;
    matrix<double> relu=lhs.maximum(0.0);
    obj["relu"]=relu;
    cache["input"]=lhs;
    result["relu"]=obj;
    result["cache"]=cache;
    return result;
}
fc_net::map1str fc_net::affine_backward(matrix<double> &upstream_grad,map1str &cache)
{
    map1str result;
    matrix<double> dx;
    matrix<double> dW;
    matrix<double> db;
    dx=upstream_grad.dot(cache["W"].t());
    dW=cache["x"].t().dot(upstream_grad);
    db=upstream_grad.sum(1);
    assert(db.axisx()==1);
    result["dx"]=dx;
    result["dW"]=dW;
    result["db"]=db;
    return result;
}

matrix<double> fc_net::relu_backward(matrix<double> &upstream_grad,map1str &cache)
{
    matrix<double> result=upstream_grad*(cache["input"]>0);

    return result;
}


fc_net::map2str fc_net::batchnorm_forward(matrix<double> &x,map1str &para,const std::string mode,const double momentum)
{
    map2str result;
    map1str output;
    map1str cache;
    assert((mode=="train")||(mode=="test"));
    if(mode=="train")
    {
        matrix<double> sample_mean=x.mean(1);
        matrix<double> sample_var=x.var(1);
        matrix<double> x_gauss=(x-sample_mean)/((sample_var+1e-5).sqrt());
        matrix<double> out=para["gamma"]*x_gauss+para["beta"];
        output["result"]=out;
        cache["x"]=x;
        cache["sample_mean"]=sample_mean;
        cache["sample_var"]=sample_var;
        cache["x_gauss"]=x_gauss;
        cache["gamma"]=para["gamma"];
        cache["beta"]=para["beta"];
        para["running_mean"] = momentum * para["running_mean"] + (1 - momentum) * para["running_mean"];
        para["running_var"] = momentum * para["running_var"] + (1 - momentum) * para["running_var"];

    }else if(mode=="test")
    {
        matrix<double> out=(x-para["running_mean"])*para["gamma"]/(para["running_var"]+1e-5)+para["beta"];
        output["result"]=out;
    }
    result["result"]=output;
    result["cache"]=cache;
    return result;
}
fc_net::map1str fc_net::batchnorm_backward(matrix<double> &upstream_grad,map1str &cache)
{
    map1str result;
    int N=upstream_grad.axisx();
    matrix<double> dx_gauss=upstream_grad*cache["gamma"];
    matrix<double> dx=(N*dx_gauss-((dx_gauss*cache["x_gauss"]).sum(1))*cache["x_gauss"]-dx_gauss.sum(1))/(cache["sample_var"]+1e-5).sqrt()/N;
    matrix<double> dgamma=(upstream_grad*cache["x_gauss"]).sum(1);
    matrix<double> dbeta=upstream_grad.sum(1);
    result["dx"]=dx;
    result["dgamma"]=dgamma;
    result["dbeta"]=dbeta;
    return result;
}

fc_net::map2str fc_net::dropout_forward(matrix<double> &x, const double p,const std::string mode)
{
    assert((mode=="train")||(mode=="test"));
    assert((p>=0)&&(p<=1));
    map2str result;
    map1str out;
    map1str cache;
    if(mode=="train")
    {
        matrix<double> temp(x.axisx(),x.axisy());
        temp.init_uniform();
        matrix<double> mask=(temp<p)*temp/p;
        out["result"]=x*mask;
        cache["mask"]=mask;
    }else if(mode=="test")
    {
        out["result"]=x;
    }
    result["result"]=out;
    result["cache"]=cache;
    return result;
}
matrix<double> fc_net::dropout_backward(matrix<double> &upstream_grad,map1str &cache)
{
    return upstream_grad*cache["mask"];
}
double fc_net::train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,
                    const double reg)
{
    std::map<std::string,map2str> cache;
    map2str affine_result;
    map2str relu_result;
    map1str relu_result1;
    relu_result1["relu"]=dataset;
    relu_result["relu"]=relu_result1;
    for(int i=1;i!=num_layers-1;++i)
    {
        affine_result=affine_forward(relu_result["relu"]["relu"],parameter["W"+num2str(i)],parameter["b"+num2str(i)]);
        relu_result=relu_forward(affine_result["affine"]["affine"]);
        cache["affine"+num2str(i)]=affine_result;
        cache["relu"+num2str(i)]=relu_result;
    }

    map2str scores=affine_forward(relu_result["relu"]["relu"],parameter["W"+num2str(num_layers-1)],parameter["b"+num2str(num_layers-1)]);

    map1str softmax_=softmax(scores["affine"]["affine"],labels,reg);

    double loss=softmax_["loss"][0][0];

    map1str affine_back_result=affine_backward(softmax_["dscores"],scores["cache"]);
    matrix<double> relu_back_result;
    parameter["W"+num2str(num_layers-1)]-=learning_rate*affine_back_result["dW"];
    parameter["W"+num2str(num_layers-1)]-=learning_rate*softmax_["dW"+num2str(num_layers-1)];
    parameter["b"+num2str(num_layers-1)]-=learning_rate*affine_back_result["db"];


    for(int i=num_layers-2;i!=0;--i)
    {
        relu_back_result=relu_backward(affine_back_result["dx"],cache["relu"+num2str(i)]["cache"]);
        affine_back_result=affine_backward(relu_back_result,cache["affine"+num2str(i)]["cache"]);
        parameter["W"+num2str(i)]-=learning_rate*affine_back_result["dW"];
        parameter["W"+num2str(i)]-=learning_rate*softmax_["dW"+num2str(i)];
        parameter["b"+num2str(i)]-=learning_rate*affine_back_result["db"];
    }

    return loss;

}
matrix<int> fc_net::predict(matrix<double> &dataset)
{

    matrix<int> result;
    map2str affine_result;
    map2str relu_result;
    map1str relu_result1;
    relu_result1["relu"]=dataset;
    relu_result["relu"]=relu_result1;
    for(int i=1;i!=num_layers-1;++i)
    {
        affine_result=affine_forward(relu_result["relu"]["relu"],parameter["W"+num2str(i)],parameter["b"+num2str(i)]);
        relu_result=relu_forward(affine_result["affine"]["affine"]);
    }

    map2str scores=affine_forward(relu_result["relu"]["relu"],parameter["W"+num2str(num_layers-1)],parameter["b"+num2str(num_layers-1)]);
    result=scores["affine"]["affine"].argmax(0);
    return result;
}





#endif // __FC_NET_H__
