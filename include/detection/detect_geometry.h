#pragma(once)
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

typedef float Ntype;
//The probability belong to face at 0
//bbxinfo contain the coordinates of bounding box at 1-4
//The bounding box regression scale at 5-8
typedef std::vector<Ntype> BBXINFO;

class DetectTools{
public:
	enum NMSType{
		UNION,
		MIN
	};
	struct isLarger {
        bool operator () ( std::vector<Ntype> & i1, std::vector<Ntype> & i2 ) {
        	return i1[4]>i2[4];
        }
    };
public:
  std::vector< std::vector<Ntype> > generateBboxes(std::vector<std::vector<Ntype> >);
  void nonmaximumSuppression(std::vector<std::vector<Ntype> >&, float , NMSType);
  std::vector<std::vector<Ntype> > bboxes2patches(std::vector<std::vector<Ntype> >&, Ntype, Ntype);
  void turn2rect(std::vector<std::vector<Ntype> >&);
  std::vector<std::vector<Ntype> > caliberateBboxes(std::vector<std::vector<Ntype> >&);
private:
};

