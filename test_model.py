from dataPreprocess import *
from model import *
from training import *
from testing import *


if __name__ == "__main__":
    load_dtProc = DataPreprocess('')
    input_size = len(load_dtProc.train_features.columns)
    target_model1 = Model(16,input_size)
    train1 = Training(target_model1.model,load_dtProc.train_features,load_dtProc.train_labels,100)
    target_model1.save_model()
    test1 = Testing(target_model1.model,load_dtProc.test_features,load_dtProc.test_labels)


    """
    target_model2 = Model(target_model1.name)
    #train2 = Training(target_model2.model,load_dtProc.train_features,load_dtProc.train_labels,100)
    test2 = Testing(target_model2.model,load_dtProc.test_features,load_dtProc.test_labels)

    test1.get_plot(train1.history)
    print(train1.history.history['mse'][-1])
    test1.get_plot_Pred_TrueVal()

    #test2.get_plot(train2.history)
    test2.get_plot_Pred_TrueVal()

    print(test1.correlation_coefficient())
    print(test2.correlation_coefficient())

    print(train1.history.history['mse'][-1])
    print(test2.get_MSE())
    """

    target_model2 = Model(32,8,input_size)
    train2 = Training(target_model2.model,load_dtProc.train_features,load_dtProc.train_labels,100)
    target_model2.save_model()
    test2 = Testing(target_model2.model,load_dtProc.test_features,load_dtProc.test_labels)


    target_model3 = Model(32,16,8,input_size)
    train3 = Training(target_model3.model,load_dtProc.train_features,load_dtProc.train_labels,100)
    target_model3.save_model()
    test3 = Testing(target_model3.model,load_dtProc.test_features,load_dtProc.test_labels)


    target_model4 = Model(32,16,8,4,input_size)
    train4 = Training(target_model4.model,load_dtProc.train_features,load_dtProc.train_labels,100)
    target_model4.save_model()
    test4 = Testing(target_model4.model,load_dtProc.test_features,load_dtProc.test_labels)

    print(test1.correlation_coefficient())
    print(test2.correlation_coefficient())
    print(test3.correlation_coefficient())
    print(test4.correlation_coefficient())

    test1.get_plot(train1.history)
    test1.get_plot_Pred_TrueVal()

    test2.get_plot(train2.history)
    test2.get_plot_Pred_TrueVal()
    

    test3.get_plot(train3.history)
    test3.get_plot_Pred_TrueVal()
   

    test4.get_plot(train4.history)
    test4.get_plot_Pred_TrueVal()

    target_model3.creat_tuned_model(train4.train_features,train4.train_labels)
    target_model4HP=target_model4.load_model(target_model3.name+'HP')
    test4HP = target_model4HP.predict(test4.test_features)
    Sum1 = 0
    Sum2 = 0

    Ymean = np.mean(test4.test_labels)

    for i in range(len(test4.test_labels)):
        Sum1 += (test4.test_labels.values[i]-test4HP[i])**2

        Sum2 += (test4.test_labels.values[i]-Ymean)**2

    R2 = float(1 - Sum1/Sum2 )
    print('HP',R2)
    a = plt.axes(aspect='equal')
    plt.scatter(test4.test_labels, test4HP,c='green')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [-4 ,4]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
   
