import pickle
import CNN_utils
import feed_forward
import back_propagation


def part_4_vectorized():
    each_cost = []
    each_acc = []
    for i in range(repeat):
        print(f"{i}th iteration")
        A3s_Ys, total_cost = back_propagation.backpropagation(train_set, n_x, n_h1, n_h2, n_y,
                                                              epochs, batch_size, alpha, vectorized)
        total_cost = [x / data_limit for x in total_cost]
        each_cost.append(total_cost)
        each_acc.append(A3s_Ys)
    CNN_utils.print_correct(each_acc)
    CNN_utils.plot_cost(epochs, each_cost)

def part_3_backpropagate():
    pickle_number = 6
    # step 3
    # 1 All same guess                      slight cost dec
    # 2 no *0.01 so 95% acc                 rising cost
    # 3 repeat of 2 - with results of 1     v nice cost dec
    # 4 repeat of 2 - med results           medium cost dec
    # 5 same as 4
    #############################
    A3s_Ys, total_cost = back_propagation.backpropagation(train_set, n_x, n_h1, n_h2, n_y,
                                                          epochs, batch_size, alpha, vectorized)
    #############################
    with open("backpropagationDump" + str(pickle_number), 'wb') as fi:
        pickle.dump((A3s_Ys, total_cost), fi)
    #############################
    with open("backpropagationDump" + str(pickle_number), 'rb') as fi:
        A3s_Ys, total_cost = pickle.load(fi)
    #############################
    total_cost = [x / data_limit for x in total_cost]
    CNN_utils.print_correct([A3s_Ys])
    CNN_utils.plot_cost(epochs, [total_cost])


def part_2_feed_forward(train_set, data_limit):
    A3s_Ys = feed_forward.just_feed_forward(train_set[:data_limit], n_x, n_h1, n_h2, n_y)
    CNN_utils.print_correct([A3s_Ys])


train_set, test_set = CNN_utils.loading_datasets()
data_limit = 200

n_x, n_h1, n_h2, n_y = 102, 150, 60, 4
# part_2_feed_forward()

epochs, batch_size, alpha, vectorized = 5, 10, 1, 0
# part_3_backpropagate()

epochs, vectorized, repeat = 20, 1, 10

part_4_vectorized()
