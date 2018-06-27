#ifndef _LIBFFM_WRAPPER_H
#define _LIBFFM_WRAPPER_H

extern "C" {

#include "ffm.h"

namespace ffm {

// new structs and methods for the wrapper

struct ffm_line {
    ffm_node* data;
    ffm_float label;
    ffm_int size;
};

struct ffm_problem {
    ffm_int size = 0;
    ffm_long num_nodes = 0;

    ffm_node* data;
    ffm_long* pos;
    ffm_float* labels;
    ffm_float* scales;

    ffm_int n = 0;
    ffm_int m = 0;
};

ffm_model ffm_load_model_c_string(char *path);

void ffm_save_model_c_string(ffm_model &model, char *path);

void ffm_init_problem(ffm_problem &p, ffm_line *data, ffm_int num_lines);

void ffm_cleanup_prediction(ffm_float *f);

void ffm_cleanup_problem(ffm_problem &p);

ffm_model ffm_init_model(ffm_problem &data, ffm_parameter params);

void ffm_copy_model(ffm_model& src, ffm_model &dest);

ffm_float ffm_train_iteration(ffm_problem &data, ffm_model &model, ffm_parameter params);

ffm_float ffm_predict_array(ffm_node *nodes, int len, ffm_model &model);

ffm_float* ffm_predict_batch(ffm_problem &data, ffm_model &model);

} // namespace ffm
}
#endif // _LIBFFM_WRAPPER_H
