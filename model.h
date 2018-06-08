#ifndef __MODEL_H__
#define __MODEL_H__
#include<ostream>
#include"fc_net.h"
#include<iomanip>
class model
{
public:
    model(const int layer[],const int num)
    {
        network=fc_net(layer,num);
    }
public:
    void train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                    const int iterations,const int min_batch,std::ostream &os,const int print_num,
                    matrix<double> &valset,matrix<int> &vallabels);
    void train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                    const int iterations,std::ostream &os,const int print_num,
                    matrix<double> &valset,matrix<int> &vallabels);
    void train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                    const int iterations,const int min_batch,std::ostream &os,const int print_num);
    void train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                    const int iterations,std::ostream &os,const int print_num);
    matrix<int> predict(matrix<double> &dataset);
    double cal_accu(matrix<int> &pred,matrix<int> &labels);
private:
    fc_net network;
};

void model::train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                  const int iterations,const int min_batch,std::ostream &os,const int print_num,
                  matrix<double> &valset,matrix<int> &vallabels)
{
    matrix<int> shuff=matrix<int>(min_batch,1);
    for(int i=0;i!=iterations;++i)
    {
        shuff.init_randint(0,dataset.axisx()-1);
        matrix<double> shuff_set=dataset.extract(shuff);
        matrix<int> shuff_labels=labels.extract(shuff);
        double loss=network.train(shuff_set,shuff_labels,learning_rate,reg);
        if((i==0)||((i+1)%print_num==0))
        {
            os<<"iterations "<<i+1<<"/"<<iterations<<": loss="<<std::setw(8)<<std::setfill('0')<<std::left<<std::showpoint<<loss<<"  ";
            matrix<int> pre_v=predict(valset);
            double acc_v=cal_accu(pre_v,vallabels);
            matrix<int> pre_t=predict(shuff_set);
            double acc_t=cal_accu(pre_t,shuff_labels);
            os<<"Accuracy of Validation set: "<<std::setw(8)<<acc_v<<"  "
                <<"Accuracy of min batch: "<<std::setw(8)<<acc_t<<"\n";
        }
    }
}
void model::train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                  const int iterations,std::ostream &os,const int print_num,
                  matrix<double> &valset,matrix<int> &vallabels)
{
    for(int i=0;i!=iterations;++i)
    {
        double loss=network.train(dataset,labels,learning_rate,reg);
        if((i==0)||((i+1)%print_num==0))
        {

            os<<"iterations "<<i+1<<"/"<<iterations<<": loss="<<std::setw(8)<<std::setfill('0')<<std::left<<std::showpoint<<loss<<"  ";
            matrix<int> pre=predict(valset);
            double acc=cal_accu(pre,vallabels);
            matrix<int> pre_t=predict(dataset);
            double acc_t=cal_accu(pre_t,labels);
            os<<"Accuracy of Validation set: "<<std::setw(8)<<acc<<"  "
                <<"Accuracy of Train set: "<<std::setw(8)<<acc_t<<"\n";
        }
    }
}
void model::train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                    const int iterations,const int min_batch,std::ostream &os,const int print_num)
{
    matrix<int> shuff=matrix<int>(min_batch,1);
    for(int i=0;i!=iterations;++i)
    {
        shuff.init_randint(0,dataset.axisx()-1);
        matrix<double> shuff_set=dataset.extract(shuff);
        matrix<int> shuff_labels=labels.extract(shuff);
        double loss=network.train(shuff_set,shuff_labels,learning_rate,reg);
        if((i==0)||((i+1)%print_num==0))
        {
            matrix<int> pre_t=predict(shuff_set);
            double acc_t=cal_accu(pre_t,shuff_labels);
            os<<"iterations "<<i+1<<"/"<<iterations<<": loss="<<std::setw(8)<<std::setfill('0')<<std::left<<std::showpoint<<loss<<"  "
                <<"Accuracy of min batch: "<<std::setw(8)<<acc_t<<"\n";
        }
    }
}
void model::train(matrix<double> &dataset,matrix<int> &labels,const double learning_rate,const double reg,
                    const int iterations,std::ostream &os,const int print_num)
{
    for(int i=0;i!=iterations;++i)
    {
        double loss=network.train(dataset,labels,learning_rate,reg);
        if((i==0)||((i+1)%print_num==0))
        {
            matrix<int> pre_t=predict(dataset);
            double acc_t=cal_accu(pre_t,labels);
            os<<"iterations "<<i+1<<"/"<<iterations<<": loss="<<std::setw(8)<<std::setfill('0')<<std::left<<std::showpoint<<loss<<"  "
                <<"Accuracy of Train set: "<<std::setw(8)<<acc_t<<"\n";
        }
    }
}


matrix<int> model::predict(matrix<double> &dataset)
{
    return network.predict(dataset);
}
double model::cal_accu(matrix<int> &pred,matrix<int> &labels)
{
    assert(pred.axisx()==labels.axisx());
    assert((pred.axisy()==1)&&(labels.axisy()==1));
    matrix<double> acc=(pred==labels);
    //std::cout<<acc;
    double result=((double)acc.sum())/((double)(pred.axisx()));
    return result;
}
#endif // __MODEL__
