#ifndef __MATRIX_H__
#define __MATRIX_H__
#include<iostream>
#include<cmath>
#include<cassert>
#include<ostream>
#include<random>

template<typename T>
class matrix
{
public:
    matrix()=default;
    explicit matrix(const int a,const int b):x(a),y(b)
    {
        mat=new T*[x];
        for(int i=0;i!=x;++i)
        {
            mat[i]=new T[y];
        }
    }
    matrix(const matrix<T> &rhs)
    {
        if(this!=&rhs)
        {
            x=rhs.x;
            y=rhs.y;
            mat=new T*[x];
            for(int i=0;i!=x;++i)
            {
                mat[i]=new T[y];
            }
            for(int i=0;i!=x;++i)
            {
                for(int j=0;j!=y;++j)
                {
                    mat[i][j]=rhs.mat[i][j];
                }
            }
        }
    }
    ~matrix()
    {
        for(int i=0;i!=x;++i)
        {
            delete [] mat[i];
        }
        delete [] mat;
    }

public:
    int axisx() const {return x;}
    int axisy() const {return y;}
    void init_zeros();
    void init_ones();
    void init_xavier();
    void init_uniform();
    void init_randint(const int lhs,const int rhs);
    matrix<T> dot(const matrix<T> &ma);
    matrix<T> sum(const int axis);
    double sum();
    matrix<T> square();
    matrix<T> sqrt();
    matrix<T> exp();
    matrix<T> log();
    matrix<T> t();
    matrix<int> argmax(const int axis);
    matrix<T> max(const int axis);
    matrix<T> maximum(const matrix<T> &rhs);
    matrix<T> maximum(const double rhs);
    matrix<int> shape();
    matrix<T> extract(const matrix<int> &rhs);
    matrix<T>& replace_rows(const int num_axis,const matrix<T>& rhs);
    matrix<T> cut(const int lhs,const int rhs);
    matrix<T>& combine(const matrix<T> &rhs);
    matrix<T>  mean(const int axis);
    T mean();
    matrix<T> var(const int axis);
    T var();
    matrix<T> std(const int axis);
    T std();

public:
    T* operator[](const int n) const
    {
        return mat[n];
    }
    matrix<T> operator-(void);
    matrix<T> operator>(const double rhs);
    matrix<T> operator>(const matrix<T> &rhs);
    matrix<T> operator<(const double rhs);
    matrix<T> operator<(const matrix<T> &rhs);
    matrix<T>& operator+=(const double rhs);
    matrix<T>& operator+=(const matrix<T> &rhs);
    matrix<T>& operator-=(const double rhs);
    matrix<T>& operator-=(const matrix<T> &rhs);
    matrix<T>& operator*=(const double rhs);
    matrix<T>& operator*=(const matrix<T>  &rhs);
    matrix<T>& operator/=(const double rhs);
    matrix<T>& operator/=(const matrix<T>  &rhs);
    matrix<T>& operator=(const matrix<T>  &rhs);
private:
    int x;
    int y;
    T **mat;
};

template<typename T>
matrix<T> operator+(const matrix<T>& lhs,const double rhs);
template<typename T>
matrix<T> operator+(const double lhs,const matrix<T>& rhs);
template<typename T>
matrix<T> operator+(const matrix<T>& lhs,const matrix<T>& rhs);
template<typename T>
matrix<T> operator-(const matrix<T>& lhs,const double rhs);
template<typename T>
matrix<T> operator-(const double lhs,const matrix<T>& rhs);
template<typename T>
matrix<T> operator-(const matrix<T>& lhs,const matrix<T>& rhs);

template<typename T>
matrix<T> operator*(const matrix<T>& lhs,const double rhs);
template<typename T>
matrix<T> operator*(const double lhs,const matrix<T>& rhs);
template<typename T>
matrix<T> operator*(const matrix<T>& lhs,const matrix<T>& rhs);
template<typename T>
matrix<T> operator/(const matrix<T>& lhs,const double rhs);
template<typename T>
matrix<T> operator/(const double lhs,const matrix<T>& rhs);
template<typename T>
matrix<T> operator/(const matrix<T>& lhs,const matrix<T>& rhs);
template<typename T>
matrix<double> operator==(const matrix<T>& lhs,const matrix<T>& rhs);
template<typename T>
std::ostream& operator<<(std::ostream& os,const matrix<T>& ma);


template<typename T>
matrix<T> operator+(const matrix<T>& lhs,const double rhs)
{
    int x=lhs.axisx();
    int y=lhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs[i][j]+rhs;
        }
    }
    return result;
}
template<typename T>
matrix<T> operator+(const double lhs,const matrix<T>& rhs)
{
    int x=rhs.axisx();
    int y=rhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs+rhs[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> operator+(const matrix<T>& lhs,const matrix<T>& rhs)
{
    int lhsx=lhs.axisx();
    int lhsy=lhs.axisy();
    int rhsx=rhs.axisx();
    int rhsy=rhs.axisy();
    int x=(lhsx>=rhsx)?lhsx:rhsx;
    int y=(lhsy>=rhsy)?lhsy:rhsy;
    matrix<T> result(x,y);
    assert(((lhsx==rhsx)&& (lhsy==rhsy))
               ||((lhsx==rhsx)&& (rhsy==1))
               ||((lhsy==rhsy)&& (rhsx==1))
               ||((lhsx==rhsx)&& (lhsy==1))
               ||((lhsy==rhsy)&& (lhsx==1)));

    if((lhsx==rhsx)&& (lhsy==rhsy))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]+rhs[i][j];
            }
        }
    }else if((lhsx==rhsx)&& (rhsy==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]+rhs[i][0];
            }
        }
    }else if((lhsy==rhsy)&& (rhsx==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]+rhs[0][j];
            }
        }
    }else if((lhsx==rhsx)&& (lhsy==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[i][0]+rhs[i][j];
            }
        }
    }else if((lhsy==rhsy)&& (lhsx==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[0][j]+rhs[i][j];
            }
        }
    }
    return result;
}

template<typename T>
matrix<T> operator-(const matrix<T>& lhs,const double rhs)
{
    int x=lhs.axisx();
    int y=lhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs[i][j]-rhs;
        }
    }
    return result;
}
template<typename T>
matrix<T> operator-(const double lhs,const matrix<T>& rhs)
{
    int x=rhs.axisx();
    int y=rhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs-rhs[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> operator-(const matrix<T>& lhs,const matrix<T>& rhs)
{
    int lhsx=lhs.axisx();
    int lhsy=lhs.axisy();
    int rhsx=rhs.axisx();
    int rhsy=rhs.axisy();
    int x=(lhsx>=rhsx)?lhsx:rhsx;
    int y=(lhsy>=rhsy)?lhsy:rhsy;
    matrix<T> result(x,y);
    assert(((lhsx==rhsx)&& (lhsy==rhsy))
               ||((lhsx==rhsx)&& (rhsy==1))
               ||((lhsy==rhsy)&& (rhsx==1))
               ||((lhsx==rhsx)&& (lhsy==1))
               ||((lhsy==rhsy)&& (lhsx==1)));

    if((lhsx==rhsx)&& (lhsy==rhsy))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]-rhs[i][j];
            }
        }
    }else if((lhsx==rhsx)&& (rhsy==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]-rhs[i][0];
            }
        }
    }else if((lhsy==rhsy)&& (rhsx==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]-rhs[0][j];
            }
        }
    }else if((lhsx==rhsx)&& (lhsy==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[i][0]-rhs[i][j];
            }
        }
    }else if((lhsy==rhsy)&& (lhsx==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[0][j]-rhs[i][j];
            }
        }
    }
    return result;
}


template<typename T>
matrix<T> operator*(const matrix<T>& lhs,const double rhs)
{
    int x=lhs.axisx();
    int y=lhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs[i][j]*rhs;
        }
    }
    return result;
}
template<typename T>
matrix<T> operator*(const double lhs,const matrix<T>& rhs)
{
    int x=rhs.axisx();
    int y=rhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs*rhs[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> operator*(const matrix<T>& lhs,const matrix<T>& rhs)
{
    int lhsx=lhs.axisx();
    int lhsy=lhs.axisy();
    int rhsx=rhs.axisx();
    int rhsy=rhs.axisy();
    int x=(lhsx>=rhsx)?lhsx:rhsx;
    int y=(lhsy>=rhsy)?lhsy:rhsy;
    matrix<T> result(x,y);
    assert(((lhsx==rhsx)&& (lhsy==rhsy))
               ||((lhsx==rhsx)&& (rhsy==1))
               ||((lhsy==rhsy)&& (rhsx==1))
               ||((lhsx==rhsx)&& (lhsy==1))
               ||((lhsy==rhsy)&& (lhsx==1)));

    if((lhsx==rhsx)&& (lhsy==rhsy))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]*rhs[i][j];
            }
        }
    }else if((lhsx==rhsx)&& (rhsy==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]*rhs[i][0];
            }
        }
    }else if((lhsy==rhsy)&& (rhsx==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]*rhs[0][j];
            }
        }
    }else if((lhsx==rhsx)&& (lhsy==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[i][0]*rhs[i][j];
            }
        }
    }else if((lhsy==rhsy)&& (lhsx==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[0][j]*rhs[i][j];
            }
        }
    }
    return result;
}

template<typename T>
matrix<T> operator/(const matrix<T>& lhs,const double rhs)
{
    int x=lhs.axisx();
    int y=lhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs[i][j]/rhs;
        }
    }
    return result;
}
template<typename T>
matrix<T> operator/(const double lhs,const matrix<T>& rhs)
{
    int x=rhs.axisx();
    int y=rhs.axisy();
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=lhs/rhs[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> operator/(const matrix<T>& lhs,const matrix<T>& rhs)
{
    int lhsx=lhs.axisx();
    int lhsy=lhs.axisy();
    int rhsx=rhs.axisx();
    int rhsy=rhs.axisy();
    int x=(lhsx>=rhsx)?lhsx:rhsx;
    int y=(lhsy>=rhsy)?lhsy:rhsy;
    matrix<T> result(x,y);
    assert(((lhsx==rhsx)&& (lhsy==rhsy))
               ||((lhsx==rhsx)&& (rhsy==1))
               ||((lhsy==rhsy)&& (rhsx==1))
               ||((lhsx==rhsx)&& (lhsy==1))
               ||((lhsy==rhsy)&& (lhsx==1)));

    if((lhsx==rhsx)&& (lhsy==rhsy))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]/rhs[i][j];
            }
        }
    }else if((lhsx==rhsx)&& (rhsy==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]/rhs[i][0];
            }
        }
    }else if((lhsy==rhsy)&& (rhsx==1))
    {
        for(int i=0;i!=lhsx;++i)
        {
            for(int j=0;j!=lhsy;++j)
            {
                result[i][j]=lhs[i][j]/rhs[0][j];
            }
        }
    }else if((lhsx==rhsx)&& (lhsy==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[i][0]/rhs[i][j];
            }
        }
    }else if((lhsy==rhsy)&& (lhsx==1))
    {
        for(int i=0;i!=rhsx;++i)
        {
            for(int j=0;j!=rhsy;++j)
            {
                result[i][j]=lhs[0][j]/rhs[i][j];
            }
        }
    }
    return result;
}
template<typename T>
std::ostream& operator<<(std::ostream& os,const matrix<T>& ma)
{

    for(int i=0;i!=ma.axisx();++i)
    {
        os<<"[";
        for(int j=0;j!=ma.axisy();++j)
        {
            os<<ma[i][j]<<", ";
        }
        os<<"]"<<"\n";
    }
    os<<"\n";
    return os;
}
template<typename T>
matrix<double> operator==(const matrix<T>& lhs,const matrix<T>& rhs)
{
    assert((lhs.axisx()==rhs.axisx())&&(lhs.axisy()==rhs.axisy()));
    matrix<double> result(lhs.axisx(),lhs.axisy());
    for(int i=0;i!=lhs.axisx();++i)
    {
        for(int j=0;j!=lhs.axisy();++j)
        {
            result[i][j]=(lhs[i][j]==rhs[i][j])+0.0;
        }
    }
    return result;
}



template<typename T>
void matrix<T>::init_zeros()
{
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]=0;
        }
    }
}
template<typename T>
void matrix<T>::init_ones()
{
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]=1;
        }
    }
}

template<typename T>
void matrix<T>::init_xavier()
{
    static std::random_device rd1;
    static std::default_random_engine engine1(rd1());
    static std::normal_distribution<double> normal(1.0,1.0);

    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]=normal(engine1)/std::sqrt(x/2);
        }
    }
}
template<typename T>
void matrix<T>::init_uniform()
{
    static std::random_device rd2;
    static std::default_random_engine engine2(rd2());
    static std::uniform_real_distribution<double> uniform(0.0,1.0);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]=uniform(engine2);
        }
    }
}
template<typename T>
void matrix<T>::init_randint(const int lhs,const int rhs)
{
    static std::random_device rd3;
    static std::default_random_engine engine3(rd3());
    static std::uniform_int_distribution<int> randint(lhs,rhs);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]=randint(engine3);
        }
    }
}
template<typename T>
matrix<T> matrix<T>::dot(const matrix<T> &ma)
{
    int ma_x=ma.axisx();
    int ma_y=ma.axisy();
    assert(y==ma_x);
    matrix<T> result(x,ma_y);
    result.init_zeros();
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=ma_y;++j)
        {
            for(int k=0;k!=ma_x;++k)
            {
                result[i][j]+=mat[i][k]*ma[k][j];
            }
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::sum(const int axis)
{

    assert((axis==0)||(axis==1));
    matrix<T> result;
    if(axis==1)
    {
        result=matrix<T>(1,y);
        result.init_zeros();
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                result[0][j]+=mat[i][j];
            }
        }

    }else if(axis==0)
    {
        result=matrix<T>(x,1);
        result.init_zeros();
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                result[i][0]+=mat[i][j];
            }
        }
    }
    return result;
}
template<typename T>
double matrix<T>::sum()
{
    double result=0;
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result+=mat[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::square()
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=mat[i][j]*mat[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::sqrt()
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=std::sqrt(mat[i][j]);
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::t()
{
    matrix<T> result(y,x);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[j][i]=mat[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::exp()
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=std::exp(mat[i][j]);
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::log()
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=std::log(mat[i][j]);
        }
    }
    return result;
}
template<typename T>
matrix<int> matrix<T>::argmax(const int axis)
{
    assert((axis==0)||(axis==1));
    matrix<int> result;
    if(axis==1)
    {
        result=matrix<int>(1,y);
        result.init_zeros();
        for(int j=0;j!=y;++j)
        {
            int index=0;
            T temp=mat[0][j];
            for(int i=1;i!=x;++i)
            {
                if(temp<mat[i][j])
                {
                    index=i;
                    temp=mat[i][j];
                }
                result[0][j]=index;
            }
        }

    }else if(axis==0)
    {
        result=matrix<int>(x,1);
        result.init_zeros();
        for(int i=0;i!=x;++i)
        {
            int index=0;
            T temp=mat[i][0];
            for(int j=1;j!=y;++j)
            {
                if(temp<mat[i][j])
                {
                    index=j;
                    temp=mat[i][j];
                }
                result[i][0]=index;
            }
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::max(const int axis)
{
    assert((axis==0)||(axis==1));
    matrix<T> result;
    if(axis==1)
    {
        result=matrix<T>(1,y);
        result.init_zeros();
        for(int j=0;j!=y;++j)
        {
            T temp=mat[0][j];
            for(int i=1;i!=x;++i)
            {
                if(temp<mat[i][j])
                {
                    temp=mat[i][j];
                }
            }
            result[0][j]=temp;
        }

    }else if(axis==0)
    {
        result=matrix<T>(x,1);
        result.init_zeros();
        for(int i=0;i!=x;++i)
        {
            T temp=mat[i][0];
            for(int j=1;j!=y;++j)
            {
                if(temp<mat[i][j])
                {
                    temp=mat[i][j];
                }
            }
            result[i][0]=temp;
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::maximum(const matrix<T> &rhs)
{
    assert((x==rhs.axisx()) && (y==rhs.axisy()));
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=((mat[i][j]>rhs[i][j])?mat[i][j]:rhs[i][j]);
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::maximum(const double rhs)
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=((mat[i][j]>rhs)?mat[i][j]:rhs);
        }
    }
    return result;
}
template<typename T>
matrix<int> matrix<T>::shape()
{
    matrix<int> result(1,2);
    result[0][0]=x;
    result[0][1]=y;
    return result;
}

template<typename T>
matrix<T> matrix<T>::extract(const matrix<int> &rhs)
{
    assert(rhs.axisy()==1);
    matrix<T> result(rhs.axisx(),y);
    for(int i=0;i!=rhs.axisx();++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=mat[rhs[i][0]][j];
        }
    }
    return result;
}
template<typename T>
matrix<T>& matrix<T>::replace_rows(const int num_axis,const matrix<T>& rhs)
{
    assert(rhs.axisx()==1);
    assert(rhs.axisy()==y);
    for(int i=0;i!=y;++i)
    {
        mat[num_axis][i]=rhs[0][i];
    }
    return *this;
}

template<typename T>
matrix<T> matrix<T>::cut(const int lhs,const int rhs)
{
    assert(rhs>lhs);
    matrix<T> result(rhs-lhs,y);
    for(int i=lhs;i!=rhs;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i-lhs][j]=mat[i][j];
        }
    }
    matrix<T> temp(x-(rhs-lhs),y);
    for(int i=0;i!=lhs;++i)
    {
        for(int j=0;j!=y;++j)
        {
            temp[i][j]=mat[i][j];
        }
    }
    for(int i=rhs;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            temp[lhs+i-rhs][j]=mat[i][j];
        }
    }
    *this=temp;
    return result;
}
template<typename T>
matrix<T>& matrix<T>::combine(const matrix<T> &rhs)
{
    assert(y==rhs.axisy());
    matrix<T> result(x+rhs.axisx(),y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=mat[i][j];
        }
    }
    for(int i=x;i!=x+rhs.axisx();++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=rhs[i-x][j];
        }
    }
    *this=result;
    return *this;
}
template<typename T>
matrix<T> matrix<T>::mean(const int axis)
{
    assert((axis==0)||(axis==1));
    matrix<T> result;
    if(axis==1)
    {
        result=this->sum(1)/x;
        assert(result.axisy()==y);

    }else if(axis==0)
    {
        result=this->sum(0)/y;
        assert(result.axisx()==x);
    }
    return result;

}
template<typename T>
T matrix<T>::mean()
{
    return this->sum()/(x*y);

}
template<typename T>
matrix<T> matrix<T>::var(const int axis)
{
    assert((axis==0)||(axis==1));
    matrix<T> result;
    if(axis==1)
    {
        result=((*this - this->mean(1)).square()).mean(1);
        assert(result.axisy()==y);

    }else if(axis==0)
    {
        result=((*this - this->mean(0)).square()).mean(0);
        assert(result.axisx()==x);
    }
    return result;
}
template<typename T>
T matrix<T>::var()
{
    return ((*this - this->mean()).square()).mean();
}
template<typename T>
matrix<T> matrix<T>::std(const int axis)
{
    assert((axis==0)||(axis==1));
    return this->var(axis).sqrt();
}
template<typename T>
T matrix<T>::std()
{
    return std::sqrt(this->var());
}

template<typename T>
matrix<T> matrix<T>::operator-(void)
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            result[i][j]=-mat[i][j];
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::operator>(const double rhs)
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            if(mat[i][j]>=rhs)
            {
                result[i][j]=1;
            }else
            {
                result[i][j]=0;
            }
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::operator>(const matrix<T> &rhs)
{
    assert((x==rhs.axisx())&& (y==rhs.axisy()));
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            if(mat[i][j]>=rhs[i][j])
            {
                result[i][j]=1;
            }else
            {
                result[i][j]=0;
            }
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::operator<(const double rhs)
{
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            if(mat[i][j]>=rhs)
            {
                result[i][j]=0;
            }else
            {
                result[i][j]=1;
            }
        }
    }
    return result;
}
template<typename T>
matrix<T> matrix<T>::operator<(const matrix<T> &rhs)
{
    assert((x==rhs.axisx())&& (y==rhs.axisy()));
    matrix<T> result(x,y);
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            if(mat[i][j]>=rhs[i][j])
            {
                result[i][j]=0;
            }else
            {
                result[i][j]=1;
            }
        }
    }
    return result;
}

template<typename T>
matrix<T>& matrix<T>::operator+=(const double rhs)
{
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]+=rhs;
        }
    }
    return *this;
}

template<typename T>
matrix<T>& matrix<T>::operator+=(const matrix<T>& rhs)
{
    assert(((x==rhs.axisx())&& (y==rhs.axisy()))
            ||(x==rhs.axisx()&& (rhs.axisy()==1))
            ||(y==rhs.axisy()&& (rhs.axisx()==1)));
    if((x==rhs.axisx())&&(y==rhs.axisy()))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]+=rhs[i][j];
            }
        }
    }else if((x==rhs.axisx()&& (rhs.axisy()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]+=rhs[i][0];
            }
        }
    }else if((y==rhs.axisy()&& (rhs.axisx()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]+=rhs[0][j];
            }
        }
    }
    return *this;
}

template<typename T>
matrix<T>& matrix<T>::operator-=(const double rhs)
{
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]-=rhs;
        }
    }
    return *this;
}
template<typename T>
matrix<T>& matrix<T>::operator-=(const matrix<T>& rhs)
{
    assert(((x==rhs.axisx())&& (y==rhs.axisy()))
            ||(x==rhs.axisx()&& (rhs.axisy()==1))
            ||(y==rhs.axisy()&& (rhs.axisx()==1)));
    if((x==rhs.axisx())&&(y==rhs.axisy()))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]-=rhs[i][j];
            }
        }
    }else if((x==rhs.axisx()&& (rhs.axisy()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]-=rhs[i][0];
            }
        }
    }else if((y==rhs.axisy()&& (rhs.axisx()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]-=rhs[0][j];
            }
        }
    }
    return *this;
}


template<typename T>
matrix<T>& matrix<T>::operator*=(const double rhs)
{
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]*=rhs;
        }
    }
    return *this;
}
template<typename T>
matrix<T>& matrix<T>::operator*=(const matrix<T>& rhs)
{
    assert(((x==rhs.axisx())&& (y==rhs.axisy()))
            ||(x==rhs.axisx()&& (rhs.axisy()==1))
            ||(y==rhs.axisy()&& (rhs.axisx()==1)));
    if((x==rhs.axisx())&&(y==rhs.axisy()))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]*=rhs[i][j];
            }
        }
    }else if((x==rhs.axisx()&& (rhs.axisy()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]*=rhs[i][0];
            }
        }
    }else if((y==rhs.axisy()&& (rhs.axisx()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]*=rhs[0][j];
            }
        }
    }
    return *this;
}

template<typename T>
matrix<T>& matrix<T>::operator/=(const double rhs)
{
    for(int i=0;i!=x;++i)
    {
        for(int j=0;j!=y;++j)
        {
            mat[i][j]/=rhs;
        }
    }
    return *this;
}
template<typename T>
matrix<T>& matrix<T>::operator/=(const matrix<T>& rhs)
{
    assert(((x==rhs.axisx())&& (y==rhs.axisy()))
            ||(x==rhs.axisx()&& (rhs.axisy()==1))
            ||(y==rhs.axisy()&& (rhs.axisx()==1)));
    if((x==rhs.axisx())&&(y==rhs.axisy()))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]/=rhs[i][j];
            }
        }
    }else if((x==rhs.axisx()&& (rhs.axisy()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]/=rhs[i][0];
            }
        }
    }else if((y==rhs.axisy()&& (rhs.axisx()==1)))
    {
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]/=rhs[0][j];
            }
        }
    }
    return *this;
}
template<typename T>
matrix<T>& matrix<T>::operator=(const matrix<T>&  rhs)
{
    if(this!=&rhs)
    {
        x=rhs.x;
        y=rhs.y;
        mat=new T*[x];
        for(int i=0;i!=x;++i)
        {
            mat[i]=new T[y];
        }
        for(int i=0;i!=x;++i)
        {
            for(int j=0;j!=y;++j)
            {
                mat[i][j]=rhs.mat[i][j];
            }
        }
    }
    return *this;
}



#endif // __MATRIX_H__
