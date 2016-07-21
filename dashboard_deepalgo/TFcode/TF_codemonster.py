import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas.io.data as web
import numpy as np
import pandas as pd

from TF_model import Model

def get_prices(symbol):
    start, end = '2007-05-02', '2016-04-11'
    data = web.DataReader(symbol, 'yahoo', start, end)
    data=pd.DataFrame(data)
    prices=data['Adj Close']
    prices=prices.astype(np.float)
    return prices

def get_returns(prices):
        return ((prices-prices.shift(-1))/prices)[:-1]
    
def get_data(list):
    l = []
    for symbol in list:
        rets = get_returns(get_prices(symbol))
        l.append(rets)
    return np.array(l).T

def sort_data(rets):
    ins = []
    outs = []
    for i in range(len(rets)-100):
        ins.append(rets[i:i+100].tolist())
        outs.append(rets[i+100])
    return np.array(ins), np.array(outs)
        
#symbol_list = ['C', 'GS']

def findata_scrape_exec(symbol_list):
    rets = get_data(symbol_list)
    ins, outs = sort_data(rets)
    ins = ins.transpose([0,2,1]).reshape([-1, len(symbol_list) * 100])
    div = int(.8 * ins.shape[0])
    train_ins, train_outs = ins[:div], outs[:div]
    test_ins, test_outs = ins[div:], outs[div:]
    #normalize inputs
    train_ins, test_ins = train_ins/np.std(ins), test_ins/np.std(ins)
    return train_ins, train_outs, test_ins, test_outs



        
        
class SmallConfig(object):
    """Small config."""
    symbol_list = ['C', 'GS']
    num_samples = 20
    input_len = 100
    n_hidden_1 = 50 
    n_hidden_2 = 50 
    learning_rate = 0.5
    
def run():
    config = SmallConfig
    train_ins, train_outs, test_ins, test_outs = findata_scrape_exec(config.symbol_list)
    if 1==1:
        sess = tf.Session()
        # initialize variables to random values
        with tf.variable_scope("model", reuse=None):
            m = Model(config = SmallConfig)
        with tf.variable_scope("model", reuse=True):
            mvalid = Model(config = SmallConfig, training=False)
        sess.run(tf.initialize_all_variables())

    # run optimizer on entire training data set many times
    train_size = train_ins.shape[0]
    for epoch in range(2000):
        start = rng.randint(train_size-50)
        batch_size = rng.randint(2,75)
        end = min(train_size, start+batch_size)

        sess.run(m.optimizer, feed_dict={m.x: train_ins[start:end], m.y_: train_outs[start:end]})
        # every 1000 iterations record progress
        if np.sqrt(epoch+1)%1== 0:
            t,s, c = sess.run([ mvalid.total_return, mvalid.ann_vol, mvalid.costfn], 
                              feed_dict={mvalid.x: train_ins, mvalid.y_: train_outs})
            t = np.mean(t)
            t = (1+t)**(1/6) -1
            s = np.mean(s)
            s = t/s
            print("Epoch:", '%04d' % (epoch+1), "cost=",c, "total return=", "{:.9f}".format(t), 
                 "sharpe=", "{:.9f}".format(s))

    trainreturn_ = sess.run([mvalid.total_return], feed_dict={mvalid.x: train_ins, mvalid.y_: train_outs})
    trainreturn_ = np.mean(trainreturn_)
    trainreturn_ = (1+trainreturn_)**(1/6) -1


    testreturn_ = sess.run([mvalid.total_return], feed_dict={mvalid.x: test_ins, mvalid.y_: test_outs})
    testreturn_ = np.mean(testreturn_)
    testreturn_ = (1+testreturn_)**(1/6) -1


    return trainreturn_, testreturn_

if __name__ == "__main__":
    run()