# -*- coding: utf-8 -*-


import pickle


def load_data_results_batched(batch_size):
    #
    file_path = "data_check_result/list_batches_result_%d.pkl" % batch_size
    #
    with open(file_path, 'rb') as fp:
        data_batches_result = pickle.load(fp)
    #
    return data_batches_result

def convert_data_batches_result(data_batches_result):
    """
    """
    data_dict = {}
    for item in data_batches_result:
        # item = batch[0], batch[1], np.argmax(results[0], -1)
        #
        x_std, y, pred = item
        #
        """
        print(x_std)
        print(y)
        print(pred)
        print()
        """
        #
        for idx in range(len(x_std)):
            
            exam_x = x_std[idx]
            exam_y = y[idx]
            exam_p = pred[idx]
                        
            x_trim = [str(tid) for tid in exam_x if tid > 0]
            #
            str_x_trim = "[S]".join(x_trim)
            # print(str_x_trim)
            #
            data_dict[str_x_trim] = ([exam_x], [exam_y]), exam_p
            #
    #
    return data_dict

#
def compare_data_batches_result(data_dict_0, data_dict_1):
    """
    """
    data_diff = {}
    #
    count_not_found = 0
    for key in data_dict_1.keys():
        if key not in data_dict_0:
            count_not_found += 1
            continue
        #
        # print(data_dict_1[key])
        # print(data_dict_0[key])
        #
        examp_p_1 = data_dict_1[key][-1]
        examp_p_0 = data_dict_0[key][-1]
        #
        if examp_p_0 == examp_p_1: continue
        #
        data_diff[key] = data_dict_0[key], data_dict_1[key]
        #       
    #
    print(count_not_found)
    #
    #
    file_path = "data_check_result/data_diff.pkl"
    with open(file_path, 'wb') as fp:
        pickle.dump(data_diff, fp)
    #
    
    return data_diff

#
if __name__ == "__main__":
    
    #
    data_result_32 = load_data_results_batched(32)
    print(len(data_result_32))
    #
    data_result_1 = load_data_results_batched(1)
    print(len(data_result_1))
    #
    
    #
    data_dict_32 = convert_data_batches_result(data_result_32)
    print(len(data_dict_32))
    #
    data_dict_1 = convert_data_batches_result(data_result_1)
    print(len(data_dict_1))
    #
    
    #
    data_diff = compare_data_batches_result(data_dict_1, data_dict_32)
    print(len(data_diff))
    #
    
    
        
    