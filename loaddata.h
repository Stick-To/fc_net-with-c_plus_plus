#ifndef __LOADDATA_H__
#define __LOADDATA_H__
#include <fstream>
#include"matrix.h"
#include<string>
#include<vector>
#include<cassert>
std::pair<matrix<int>,matrix<double>> loadData(const int num_data,const int num_attributes,const std::string& file_name);
std::vector<std::string> split(const std::string str,const char s);
std::pair<matrix<int>,matrix<double>> str2num(std::vector<std::string> str_vec,const int num_attributes);

std::pair<matrix<int>,matrix<double>> loadData(const int num_data,const int num_attributes,const std::string& file_name)
{
    matrix<int> labels=matrix<int>(num_data,1);
    matrix<double> attributes=matrix<double>(num_data,num_attributes);
    std::pair<matrix<int>,matrix<double>> result;
    std::string line;
	std::ifstream file(file_name);
	assert(file);
	for (int j = 0; j != num_data; ++j)
	{
		getline(file, line);
		std::vector<std::string> str_vec=split(line,',');
		assert(str_vec.size()==num_attributes+1);
        std::pair<matrix<int>,matrix<double>> temp=str2num(str_vec,num_attributes);
        labels.replace_rows(j,temp.first);
        attributes.replace_rows(j,temp.second);
	}
    result.first=labels;
    result.second=attributes;
    return result;
}
std::vector<std::string> split(const std::string str,const char s)
{
    std::vector<std::string> result;
    std::string temp="";
    for(auto iter=str.begin();iter!=str.end();++iter)
    {
        if(*iter!=s)
        {
            temp+=*iter;
        }else
        {
            result.push_back(temp);
            temp="";
        }
    }
    result.push_back(temp);
    return result;

}
std::pair<matrix<int>,matrix<double>> str2num(std::vector<std::string> str_vec,const int num_attributes)
{

    matrix<int> label=matrix<int>(1,1);
    label.init_zeros();
    std::string str=str_vec[0];
    matrix<double> d_mat=matrix<double>(1,num_attributes);
    d_mat.init_zeros();
    for(auto iter=str.begin();iter!=str.end();++iter)
    {
        if (*iter>=48 && *iter<=57)
		{
			label[0][0] = label[0][0]* 10 + ((int)(*iter) - 48);
		}
    }
    for(auto iter=str_vec.begin()+1;iter!=str_vec.end();++iter)
    {
        int first=0;
        double second=0;
        int flag=0;
        int temp=1;
        for(auto siter=iter->begin();siter!=iter->end();++siter)
        {
            if(*siter=='.')
            {
                flag=1;
            }
            if(flag==0)
            {
                if (*siter>=48 && *siter<=57)
                {
                    first = first * 10 + ((int)(*siter) - 48);
                }
            }else if(flag==1)
            {
                if (*siter>=48 && *siter<=57)
                {
                    second +=(((int)(*siter) - 48)+0.0)/(pow(10,temp));
                    temp+=1;
                }
            }
        }
        d_mat[0][iter-str_vec.begin()-1]=(first+second);
    }
    std::pair<matrix<int>,matrix<double>> result;
    result.first=label;
    result.second=d_mat;
    return result;
}
#endif // __MYTYRE_H__
