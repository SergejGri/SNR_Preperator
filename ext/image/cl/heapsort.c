//Copyright 2015-2020 University Wuerzburg.

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

int iParent(int i){
    return ((int)((i-1) / 2));
    }
int iLeftChild(int i){
    return (2*i + 1);
    }
int iRightChild(int i){
    return (2*i + 2);
    }

void swap_vals(float* data, int k_1, int k_2){
    float val = data[k_1];
    data[k_1] = data[k_2];
    data[k_2] = val;
    }

void siftDown(float* data, int start, int end){
    int root = start;
    int swap, child;

    while (iLeftChild(root) <= end){
        child = iLeftChild(root);
        swap = root;

        if (data[swap] < data[child]){swap = child;}
        if (child+1 <= end && data[swap] < data[child+1]){swap = child + 1;}
        if (swap == root){return;}
        else{
            swap_vals(data, root, swap);
            root = swap;}
        }
    }

void heapify(float* data, int count){
    int start = iParent(count-1);

    while (start >= 0){
        siftDown(data, start, count - 1);
        start -= 1;
        }
    }

void heapsort(float* data, int count){
    heapify(data, count);
    for (int end=count-1; end>0; end--){
        swap_vals(data, end, 0);
        siftDown(data, 0, end-1);
        }
    }

int number_smaller_entries(float* data, float val_insert, int left, int right){
    int num_smaller = (right+left)/2;
    while (num_smaller != left){
        if (val_insert > data[num_smaller]){left  = num_smaller;}
        else                               {right = num_smaller;}
        num_smaller = (right+left)/2;
        }
    return (num_smaller + (val_insert > data[left]) + (val_insert > data[right]));
    }

void update_sorted(float* data, int count, float val_insert, float val_remove){
    // note: if val_remove is not in data and larger than all values, this function will crash the kernel
    // catching this error wouly slow down computing [code: ]
    int pos_remove_index = number_smaller_entries(data, val_remove, 0, count-1);
    pos_remove_index = min(pos_remove_index, count-1);
    int pos_insert_index = number_smaller_entries(data, val_insert, 0, count-1);
    pos_insert_index = min(pos_insert_index, count-1);
    pos_insert_index -= (pos_insert_index > pos_remove_index)*(val_insert < data[pos_insert_index]);

    // is twice as fast with the if-clause instead of positive/negative stepping
    if (pos_remove_index <= pos_insert_index){ // move old values down
        for (int j=pos_remove_index; j<pos_insert_index; j++){data[j] = data[j+1];}}

    else                                     { // move old values up
        for (int i=pos_remove_index; i>pos_insert_index; i--){data[i] = data[i-1];}}
    data[pos_insert_index] = val_insert;
    }

 
