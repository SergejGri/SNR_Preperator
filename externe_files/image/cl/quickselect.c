//Copyright 2015-2020 University Wuerzburg.

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
void swap(float* values, int k, int j){
    float val = values[k];
    values[k] = values[j];
    values[j] = val;
    }

void select_pivot(float* values, int left, int right){
    int mid = (left + right) / 2;
    if (values[mid] < values[left]){swap(values, left, mid);}
    if (values[right] < values[left]){swap(values, left, right);}
    if (values[mid] < values[right]){swap(values, mid, right);}
}

float quickselect(float* values, int left, int right, int k){
    if (left==right){return(values[left]);}
    int fill_index=0;
    float pivot_val, swap_val;

    while (k != fill_index){
        select_pivot(values, left, right);
        pivot_val = values[right];
        fill_index = left;
        for (int test_index=left; test_index<right; test_index++){
            if (values[test_index] < pivot_val){
                swap(values, fill_index, test_index);
                fill_index++;
                }
            }
        swap(values, fill_index, right);

        if (k < fill_index){right = fill_index-1;}
        else{               left  = fill_index+1;}
        }
    return values[k];
    }

float quickselect_median(float* values, int length){
    return(quickselect(values, 0, length-1, length / 2));
    }

