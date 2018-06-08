#include"model.h"
#include"loaddata.h"
using namespace std;
typedef pair<matrix<int>,matrix<double>> pairmm;
int main()
{
    string file_name=".\\wine.data";
    pairmm data=loadData(178,13,file_name);
    matrix<double> trainset=data.second;
    trainset/=trainset.max(1);
    matrix<int> trainlabel=data.first-1;

    matrix<double> valset=trainset.cut(0,10);
    matrix<int> vallabel=trainlabel.cut(0,10);
    valset.combine(trainset.cut(100,110));
    vallabel.combine(trainlabel.cut(100,110));
    valset.combine(trainset.cut(140,150));
    vallabel.combine(trainlabel.cut(140,150));

    int layer[]={13,7,3};
    model m=model(layer,3);


    m.train(trainset,trainlabel,1e-2,1e-3,10000,50,std::cout,100,valset,vallabel);

}
