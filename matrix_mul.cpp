#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


#include <vector>
#include <ctime>
#include <iostream>
#include<fstream>
#include <CL/sycl.hpp>
using namespace sycl;

std::vector<std::vector<int>> mul(std::vector<std::vector<int>>& a, std::vector<std::vector<int>>&b)
{
	int m = a.size();
	int n = a[0].size();
	int r = b.size();
	std::cout << m <<"  "<< n << "  "<<r<<'\n';

	std::vector<std::vector<int>> res(m);
	for (auto it = res.begin(); it < res.end(); ++it) {
		it->resize(r, 0);
	}
	
	for (auto it = res.begin(); it != res.end(); ++it) {
		for (auto it1 = it->begin(); it1 !=it->end(); ++it1) {
			*it1 = 0;
		}
	}
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < r; ++j) {
			for (int k = 0; k < n; ++k) {
				res[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return res;
}

void generate_matrix(std::vector<std::vector<int>>& a) {
	srand(time(NULL) + rand());
	for (auto it = a.begin(); it != a.end(); ++it) {
		for (auto it1 = it->begin(); it1 != it->end(); ++it1) {
			*it1 = (int)rand() % 20;
		}
	}
}


void mul1(queue &q,std::vector<int> &a,std::vector<int> &b,std::vector<int> &c,int m,int n,int r){
    auto R= range(m*n*r);
    buffer buf1(a);
    buffer buf2(b);
    buffer buf3(c);
    q.submit([&](auto &h){
        accessor V1(buf1,h,read_only);
        accessor V2(buf2,h,read_only);
        accessor V3(buf3,h,write_only);
        h.parallel_for(R,[=](auto i){
            V3[i]=V1[(i/n/r*n)+i%n]*V2[((i%n)*r)+i/n/r];
        });
    });
    q.wait_and_throw();
}

void sum1(queue &q,std::vector<int>&a, std::vector<int> out,int m,int n,int r ){
    auto R= range(m*r);
    buffer buf1(a);
    buffer buf2(out);
    q.submit([&](auto &h){
        accessor V1(buf1,h,read_only);
        accessor V2(buf2,h,write_only);
        h.parallel_for(R,[=](auto i){
            for(int j=0;j<n;++j){
                V2[i]+=V1[i*n+j];
            }
        });
    });
    q.wait_and_throw();
}

int M = 1000;
int N = 1000;
int R = 1000;
clock_t start, end;

int main() {
    
    std::vector<std::vector<int>> a(M);

	for (auto it = a.begin(); it < a.end(); ++it) {
		it->resize(N, 1);
	}
	std::vector<std::vector<int>> b(N);
	for (auto it = b.begin(); it < b.end(); ++it) {
		it->resize(R, 1);
	}
	std::vector<std::vector<int>> c(M);
	for (auto it = c.begin(); it < c.end(); ++it) {
		it->resize(R, 0);
	}
	//generate_matrix(a);
	//generate_matrix(b);
	

	start = clock();		//程序开始计时
	c=mul(a, b);

	end = clock();		//程序结束用时
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	//std::cout << "Total time0:" << endtime << std::endl;		//s为单位
	//std::cout << "Total time0:" << endtime * 1000 << "ms" << std::endl;	//ms为单位
    
    std::fstream f;
	f.open("data.txt",std::ios::out);
	//输入你想写入的内容 
	f<< "Total time0:" << endtime  << "s" << std::endl;
	
	
	
    
    //gpu_selector seletor;
    queue q(gpu_selector_v);
    std::cout << "Device: "
    << q.get_device().get_info<info::device::name>() << std::endl;
    
    std::vector<int> a1(N * M);
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			a1[i * N + j] = a[i][j];
		}
	}

	std::vector<int> b1(N*R);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < R; ++j) {
			b1[i * N + j] = b[i][j];
		}
	}

	std::vector<int> c1(N * M*R);
	std::vector<int> c2(M * R);
    
    start = clock();
    
    //mul1(q,a1,b1,c1,M,N,R);
    
    //sum1(q,c1,c2, M, N,R);
    
    end = clock();
    
    endtime = (double)(end - start) / CLOCKS_PER_SEC;
    //std::cout << "Total time1:" << endtime * 1000 << "ms" << std::endl;
     
	f<< "Total time1:" << endtime << "s" << std::endl;
    f.close();

}
