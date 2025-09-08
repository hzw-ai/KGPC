import json
import pickle

import pandas as pd
import torch
from matplotlib import pyplot as plt

from utils import *

os.environ['HOME'] = '/python_work/KGPC/'
from src.spodernet.spodernet.preprocessing.pipeline import DatasetStreamer
from src.spodernet.spodernet.preprocessing.processors import JsonLoaderProcessors, AddToVocab, \
    StreamToHDF5, CustomTokenizer
from src.spodernet.spodernet.preprocessing.processors import ConvertTokenToIdx, ToLower, \
    DictKey2ListMapper
from src.spodernet.spodernet.preprocessing.batching import StreamBatcher
from src.spodernet.spodernet.preprocessing.pipeline import Pipeline
from src.spodernet.spodernet.hooks import LossHook, ETAHook
from src.spodernet.spodernet.preprocessing.processors import TargetIdx2MultiTarget
import scipy.sparse as sp
from models import SACN
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.numpy().astype(np.int64)
    # y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1'  # entities
    keys2keys['rel'] = 'rel'  # relations
    keys2keys['rel_eval'] = 'rel'  # relations
    keys2keys['e2'] = 'e1'  # entities
    keys2keys['e2_multi1'] = 'e1'  # entity
    keys2keys['e2_multi2'] = 'e1'  # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(args['dataset'], delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys),
                         keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
    p.add_post_processor(StreamToHDF5('full', samples_per_file=1000, keys=input_keys))

    p.execute(d)
    p.save_vocabs()

    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([full_path, train_path, dev_ranking_path, test_ranking_path],
                          ['full', 'train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys),
                             keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)


def main(args):
    if args['process']: preprocess(args['dataset'], delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args['dataset'], keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']
    node_list = p.state['vocab']['e1']
    rel_list = p.state['vocab']['rel']

    num_entities = vocab['e1'].num_token
    num_relations = vocab['rel'].num_token

    all_token_attr = [[] for i in range(num_entities)]
    with open('data/{}/e1rel_to_e2_full.json'.format(args['dataset']), 'rt', encoding='UTF-8') as f:
        for line in f.readlines():
            cur_data = json.loads(line)
            cur_token = cur_data['e1'].lower()
            cur_id = node_list.get_idx(cur_token)
            cur_attr = all_token_attr[cur_id]
            attr = [node_list.get_idx(token.lower()) for token in cur_data['e2_multi1'].split(' ')]
            cur_attr.extend(attr)
            all_token_attr[cur_id] = cur_attr
    # 用0填充缺失的值
    max_len = max(len(row) for row in all_token_attr)
    padded_list = [row + [0] * (max_len - len(row)) for row in all_token_attr]
    all_token_attr = torch.LongTensor(padded_list)

    patent_triple = pd.read_csv('data/{}/patent_triple.csv'.format(args['dataset']))
    patent_triple = patent_triple[patent_triple['relation'] ==
                                  args['label_relation'] if args['label_relation'] else 'has_grant'].values.tolist()
    patent_grant = {i[0]: i[2] for i in patent_triple}
    patent_index = []
    labels = []
    with open('data/%s/embeddings_bge.pkl' % args['dataset'], 'rb') as f:
        embeddings = pickle.load(f)
    patent_embs = []
    for i in range(num_entities):
        token = node_list.get_word(i)
        if '@' not in token:
            continue
        patent_index.append(i)
        patent_embs.append(torch.tensor(embeddings[token.upper()]).to(args['device']))
        labels.append(int(patent_grant[token.upper()]))
    labels = torch.LongTensor(labels)
    patent_embs = torch.cat(patent_embs, dim=0)
    entity_emb = None

    print('Finished the preprocessing')

    # full_batcher = StreamBatcher(args['dataset'], 'full', 1, randomize=True, keys=input_keys)
    # full_batcher = [item for item in full_batcher]

    # dev_rank_batcher = StreamBatcher(args['dataset'], 'dev_ranking', args['batch_size'], randomize=False,
    #                                  loader_threads=4, keys=input_keys)
    # test_rank_batcher = StreamBatcher(args['dataset'], 'test_ranking', args['batch_size'], randomize=False,
    #                                   loader_threads=4, keys=input_keys)

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    n_clusters = int(args['n_clusters'])

    X = torch.LongTensor([i for i in range(num_entities)])

    model = SACN(num_entities, num_relations, **args)
    model = model.to(args['device'])
    X = X.to(args['device'])

    if load:
        model_params = torch.load(model_path)
        model.load_state_dict(model_params)
        model.eval()
    else:
        model.init()
        model.set_emb(patent_index, patent_embs)
        train_batcher = StreamBatcher(args['dataset'], 'train', args['batch_size'], randomize=False, keys=input_keys)
        train_batcher.at_batch_prepared_observers.insert(1, TargetIdx2MultiTarget(num_entities, 'e2_multi1',
                                                                                  'e2_multi1_binary'))
        data = []
        rows = []
        columns = []

        for i, str2var in enumerate(train_batcher):
            if i % 10 == 0: print("batch number:", i)
            for j in range(str2var['e1'].shape[0]):
                for k in range(str2var['e2_multi1'][j].shape[0]):
                    if str2var['e2_multi1'][j][k] != 0:
                        # a = str2var['rel'][j].cpu()
                        data.append(str2var['rel'][j].cpu())
                        rows.append(str2var['e1'][j].cpu().tolist()[0])
                        columns.append(str2var['e2_multi1'][j][k].cpu())
                    else:
                        break

        rows = rows + [i for i in range(num_entities)]
        columns = columns + [i for i in range(num_entities)]
        data = data + [num_relations for i in range(num_entities)]

        indices = torch.LongTensor([rows, columns]).cuda()
        v = torch.LongTensor(data).cuda()
        adjacencies = [indices, v, num_entities]

        del data, rows, columns

        train_batcher = StreamBatcher(args['dataset'], 'train', args['batch_size'], randomize=False, keys=input_keys)

        eta = ETAHook('train', print_every_x_batches=100)
        train_batcher.subscribe_to_events(eta)
        train_batcher.subscribe_to_start_of_epoch_event(eta)
        train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=10))

        train_batcher.at_batch_prepared_observers.insert(1, TargetIdx2MultiTarget(num_entities, 'e2_multi1',
                                                                                  'e2_multi1_binary'))
        opt = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        for epoch in range(args['num_epochs']):
            model.train()
            total_loss = 0.0
            total_count = 0
            for i, str2var in tqdm(enumerate(train_batcher)):
                opt.zero_grad()
                e1 = str2var['e1'].cuda()
                rel = str2var['rel'].cuda()
                attr = all_token_attr[e1].cuda()

                e2_multi = str2var['e2_multi1_binary'].float().cuda()
                # label smoothing
                e2_multi = ((1.0 - args['label_smoothing_epsilon']) * e2_multi) + (1.0 / e2_multi.size(1))
                pred, logit = model.forward(e1, rel, attr, X, adjacencies)
                # loss = kl_loss
                loss = model.loss(pred, e2_multi)
                loss += model.kl_loss(F.softmax(logit, dim=-1))
                loss.backward()
                opt.step()
                train_batcher.state.loss = loss.cpu()
                total_loss += loss.item()
                total_count += 1

            model.eval()
            entity_emb = model.emb_e.weight
            patent_emb = entity_emb[patent_index].cpu().detach().numpy()
            patent_emb = StandardScaler().fit_transform(patent_emb)

            kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
            y_pred = kmeans.fit_predict(patent_emb)

            acc = cluster_acc(labels, y_pred)
            nmi = nmi_score(labels, y_pred)
            ari = ari_score(labels, y_pred)
            print('===== Clustering performance: =====')
            result = 'epoch {} Acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(epoch, acc, nmi, ari)
            print(result)

            with open(cur_save_log_path + 'result.txt', 'at') as f:
                f.write(result + '\n')

        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

    # ---------- 聚类任务 ------------ #
    entity_emb = model.emb_e.weight
    # entity_emb = embedding
    patent_emb = entity_emb[patent_index].cpu().detach().numpy()
    patent_emb = StandardScaler().fit_transform(patent_emb)

    acc_list, nmi_list, ari_list = [], [], []
    for kmeans_random_state in range(10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state, n_init='auto')
        y_pred = kmeans.fit_predict(patent_emb)
        cluster_centers = kmeans.cluster_centers_

        acc = cluster_acc(labels, y_pred)
        nmi = nmi_score(labels, y_pred)
        ari = ari_score(labels, y_pred)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
    print('===== Clustering performance: =====')
    # result = 'Acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari)
    result = ('Acc [{:.4f}, {:.4f}], nmi [{:.4f}, {:.4f}], ari [{:.4f}, {:.4f}]'
			  .format(np.mean(acc_list), np.std(acc_list), np.mean(nmi_list), np.std(nmi_list),
					  np.mean(ari_list), np.std(ari_list)))
    print(result)
    if not load:
        with open(cur_save_log_path + 'result.txt', 'at') as f:
            f.write(result)

    # kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    # y_pred = kmeans.fit_predict(patent_emb)
    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(patent_emb)
    # plt.rcParams['font.family'] = 'Microsoft YaHei'
    # marker_list = ['o', '^', 'D']
    # for cluster_label in range(n_clusters):
    #     # plt.scatter(X_tsne[y_pred == cluster_label, 0], X_tsne[y_pred == cluster_label, 1], label=f'Cluster {cluster_label}')
    #     plt.scatter(X_tsne[y_pred == cluster_label, 0], X_tsne[y_pred == cluster_label, 1],
    #                 label=f'第 {cluster_label} 个簇', s=15, marker=marker_list[cluster_label])
    # plt.legend()
    # # plt.show()
    # plt.savefig(cur_save_log_path + "{}_0.pdf".format(args['num_epochs']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SGC')
    parser.add_argument('-d', '--dataset', type=str, default='patent1270',
                        help='dataset name')
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--process', type=bool, default=True, help='dataset process')
    parser.add_argument('--patience', type=int, help='stop patience')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--channels', type=int)
    parser.add_argument('--kernel_size', type=int)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--init_emb_size', type=int)
    parser.add_argument('--gc1_emb_size', type=int)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--label_smoothing_epsilon', type=int)

    args = parser.parse_args().__dict__
    args = setup(args)

    configs = yaml.load(
        open(f'config/{args["dataset"]}.yaml'),
        Loader=yaml.FullLoader
    )

    args_temp = dict(**configs)
    for key in args:
        value = args[key]
        if value is not None:
            args_temp[key] = value
    args = args_temp

    dt = datetime.datetime.now()
    cur_save_log_path = 'logs/{}/{}-{:02d}-{:02d}-{:02d}/'.format(args['dataset'], dt.date(), dt.hour, dt.minute,
                                                                  dt.second)
    load = False
    if not load:
        mkdir_p(cur_save_log_path)
    args['log-dir'] = cur_save_log_path

    model_path = cur_save_log_path + 'best_model.pth'

    if not load:
        # 保存配置
        with open(cur_save_log_path + 'config.json', 'wt') as f:
            json.dump(args, f)

    main(args)
