import pickle
import CNN_utils
import feed_forward
import back_propagation

def backpropagate_and_pickle():
    A3s_Ys, total_cost = back_propagation.backpropagation(train_set, n_x, n_h1, n_h2, n_y,
                                                          epochs, batch_size, alpha, vectorized)
    with open("backpropagationDump" + str(pickle_number), 'wb') as fi:
        pickle.dump((A3s_Ys, total_cost), fi)

def feed_forward_test(train_set, data_limit):
    A3s_Ys = feed_forward.just_feed_forward(train_set[:data_limit], n_x, n_h1, n_h2, n_y)
    CNN_utils.print_correct_percentage(A3s_Ys)


train_set, test_set = CNN_utils.loading_datasets()
data_limit = 200

n_x, n_h1, n_h2, n_y = 102, 150, 60, 4
# feed_forward_test(train_set, data_limit)

epochs, batch_size, alpha, vectorized = 20, 10, 1, 1
pickle_number = 6
# step 3
# 1 All same guess                      slight cost dec
# 2 no *0.01 so 95% acc                 rising cost
# 3 repeat of 2 - with results of 1     v nice cost dec
# 4 repeat of 2 - med results           medium cost dec
# 5 same as 4

backpropagate_and_pickle()

with open("backpropagationDump" + str(pickle_number), 'rb') as fi:
    A3s_Ys, total_cost = pickle.load(fi)
total_cost = [x/data_limit for x in total_cost]
CNN_utils.print_correct_percentage(A3s_Ys)
CNN_utils.plot_costs_per_epoch(epochs, total_cost)
