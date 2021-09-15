import matplotlib.pyplot as plt


def compare_same_model(modelname):
    if modelname != "":
        modelname += "_"

    prefix = "modelfiles/textclassify_eb_" + modelname

    with open(prefix + "bce/history.txt", "r", encoding="utf-8") as fr:
        history_bce = fr.read()
        history_bce = eval(history_bce)

    with open(prefix + "bcew/history.txt", "r", encoding="utf-8") as fr:
        history_bcew = fr.read()
        history_bcew = eval(history_bcew)

    with open(prefix + "fl/history.txt", "r", encoding="utf-8") as fr:
        history_fl = fr.read()
        history_fl = eval(history_fl)

    with open(prefix + "flw/history.txt", "r", encoding="utf-8") as fr:
        history_flw = fr.read()
        history_flw = eval(history_flw)

    plt.subplot(221)
    plt.plot(history_bce["acc"])
    plt.plot(history_bcew["acc"])
    plt.plot(history_fl["acc"])
    plt.plot(history_flw["acc"])
    plt.title('acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew', 'fl', 'flw'], loc='best', prop={'size': 4})

    plt.subplot(222)
    plt.plot(history_bce["F1"])
    plt.plot(history_bcew["F1"])
    plt.plot(history_fl["F1"])
    plt.plot(history_flw["F1"])
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew', 'fl', 'flw'], loc='best', prop={'size': 4})

    plt.subplot(223)
    plt.plot(history_bce["val_acc"])
    plt.plot(history_bcew["val_acc"])
    plt.plot(history_fl["val_acc"])
    plt.plot(history_flw["val_acc"])
    plt.title('val_acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew', 'fl', 'flw'], loc='best', prop={'size': 4})

    plt.subplot(224)
    plt.plot(history_bce["val_F1"])
    plt.plot(history_bcew["val_F1"])
    plt.plot(history_fl["val_F1"])
    plt.plot(history_flw["val_F1"])
    plt.title('val_F1')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew', 'fl', 'flw'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("compare_model_" + modelname[:-1] + ".jpg", dpi=500, bbox_inches="tight")
    # plt.show()


def compare_fasttext():
    modelname = "fasttext_"

    prefix = "modelfiles/textclassify_eb_" + modelname

    with open(prefix + "bce/history.txt", "r", encoding="utf-8") as fr:
        history_bce = fr.read()
        history_bce = eval(history_bce)

    with open(prefix + "bcew1/history.txt", "r", encoding="utf-8") as fr:
        history_bcew1 = fr.read()
        history_bcew1 = eval(history_bcew1)

    with open(prefix + "bcew2/history.txt", "r", encoding="utf-8") as fr:
        history_bcew2 = fr.read()
        history_bcew2 = eval(history_bcew2)

    with open(prefix + "fl/history.txt", "r", encoding="utf-8") as fr:
        history_fl = fr.read()
        history_fl = eval(history_fl)

    with open(prefix + "flw1/history.txt", "r", encoding="utf-8") as fr:
        history_flw1 = fr.read()
        history_flw1 = eval(history_flw1)

    with open(prefix + "flw2/history.txt", "r", encoding="utf-8") as fr:
        history_flw2 = fr.read()
        history_flw2 = eval(history_flw2)

    plt.subplot(221)
    plt.plot(history_bce["acc"])
    plt.plot(history_bcew1["acc"])
    plt.plot(history_bcew2["acc"])
    plt.plot(history_fl["acc"])
    plt.plot(history_flw1["acc"])
    plt.plot(history_flw2["acc"])
    plt.title('acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew1', 'bcew2', 'fl', 'flw1', 'flw2'], loc='best', prop={'size': 4})

    plt.subplot(222)
    plt.plot(history_bce["F1"])
    plt.plot(history_bcew1["F1"])
    plt.plot(history_bcew2["F1"])
    plt.plot(history_fl["F1"])
    plt.plot(history_flw1["F1"])
    plt.plot(history_flw2["F1"])
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew1', 'bcew2', 'fl', 'flw1', 'flw2'], loc='best', prop={'size': 4})

    plt.subplot(223)
    plt.plot(history_bce["val_acc"])
    plt.plot(history_bcew1["val_acc"])
    plt.plot(history_bcew2["val_acc"])
    plt.plot(history_fl["val_acc"])
    plt.plot(history_flw1["val_acc"])
    plt.plot(history_flw2["val_acc"])
    plt.title('val_acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew1', 'bcew2', 'fl', 'flw1', 'flw2'], loc='best', prop={'size': 4})

    plt.subplot(224)
    plt.plot(history_bce["val_F1"])
    plt.plot(history_bcew1["val_F1"])
    plt.plot(history_bcew2["val_F1"])
    plt.plot(history_fl["val_F1"])
    plt.plot(history_flw1["val_F1"])
    plt.plot(history_flw2["val_F1"])
    plt.title('val_F1')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['bce', 'bcew1', 'bcew2', 'fl', 'flw1', 'flw2'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("compare_model_" + modelname[:-1] + ".jpg", dpi=500, bbox_inches="tight")
    # plt.show()


def compare_gamma():
    modelname = "tta_flw_gamma"

    prefix = "modelfiles/textclassify_eb_" + modelname

    with open(prefix + "2/history.txt", "r", encoding="utf-8") as fr:
        history2 = fr.read()
        history2 = eval(history2)
    with open(prefix + "3/history.txt", "r", encoding="utf-8") as fr:
        history3 = fr.read()
        history3 = eval(history3)
    with open(prefix + "4/history.txt", "r", encoding="utf-8") as fr:
        history4 = fr.read()
        history4 = eval(history4)
    with open(prefix + "5/history.txt", "r", encoding="utf-8") as fr:
        history5 = fr.read()
        history5 = eval(history5)
    with open(prefix + "10/history.txt", "r", encoding="utf-8") as fr:
        history10 = fr.read()
        history10 = eval(history10)
    with open(prefix + "15/history.txt", "r", encoding="utf-8") as fr:
        history15 = fr.read()
        history15 = eval(history15)
    with open(prefix + "20/history.txt", "r", encoding="utf-8") as fr:
        history20 = fr.read()
        history20 = eval(history20)

    plt.subplot(221)
    metrics = "acc"
    plt.plot(history2[metrics])
    plt.plot(history3[metrics])
    plt.plot(history4[metrics])
    plt.plot(history5[metrics])
    plt.plot(history10[metrics])
    plt.plot(history15[metrics])
    plt.plot(history20[metrics])
    plt.title(metrics)
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['2', '3', '4', '5', '10', '15', '20'], loc='best', prop={'size': 4})

    plt.subplot(222)
    metrics = "F1"
    plt.plot(history2[metrics])
    plt.plot(history3[metrics])
    plt.plot(history4[metrics])
    plt.plot(history5[metrics])
    plt.plot(history10[metrics])
    plt.plot(history15[metrics])
    plt.plot(history20[metrics])
    plt.title(metrics)
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['2', '3', '4', '5', '10', '15', '20'], loc='best', prop={'size': 4})

    plt.subplot(223)
    metrics = "val_acc"
    plt.plot(history2[metrics])
    plt.plot(history3[metrics])
    plt.plot(history4[metrics])
    plt.plot(history5[metrics])
    plt.plot(history10[metrics])
    plt.plot(history15[metrics])
    plt.plot(history20[metrics])
    plt.title(metrics)
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['2', '3', '4', '5', '10', '15', '20'], loc='best', prop={'size': 4})

    plt.subplot(224)
    metrics = "val_F1"
    plt.plot(history2[metrics])
    plt.plot(history3[metrics])
    plt.plot(history4[metrics])
    plt.plot(history5[metrics])
    plt.plot(history10[metrics])
    plt.plot(history15[metrics])
    plt.plot(history20[metrics])
    plt.title(metrics)
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['2', '3', '4', '5', '10', '15', '20'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("compare_gamma_tta_flw.jpg", dpi=500, bbox_inches="tight")
    # plt.show()


def compare_same_loss(lossname):
    model = ["", "attn_", "GRU_attn_", "bert_", "tta_"]

    history = []
    for i in range(len(model)):
        prefix = "modelfiles/textclassify_eb_" + model[i]

        with open(prefix + lossname + "/history.txt", "r", encoding="utf-8") as fr:
            his = fr.read()
            history.append(eval(his))

    plt.subplot(221)
    for i in range(len(model)):
        plt.plot(history[i]["acc"])
    plt.title('acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(["textcnn", "textcnn_attn", "GRU_attn", "bert", "tta"], loc='best', prop={'size': 4})

    plt.subplot(222)
    for i in range(len(model)):
        plt.plot(history[i]["F1"])
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(["textcnn", "textcnn_attn", "GRU_attn", "bert", "tta"], loc='best', prop={'size': 4})

    plt.subplot(223)
    for i in range(len(model)):
        plt.plot(history[i]["val_acc"])
    plt.title('val_acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(["textcnn", "textcnn_attn", "GRU_attn", "bert", "tta"], loc='best', prop={'size': 4})

    plt.subplot(224)
    for i in range(len(model)):
        plt.plot(history[i]["val_F1"])
    plt.title('val_F1')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(["textcnn", "textcnn_attn", "GRU_attn", "bert", "tta"], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics with Loss=" + lossname)

    plt.tight_layout()
    plt.savefig("compare_loss_" + lossname + ".jpg", dpi=500, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # compare_same_model("fastformer")

    # compare_same_loss("flw")

    # compare_fasttext()

    compare_gamma()
