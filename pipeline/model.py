import logging

import coloredlogs

#from module import cluster
from module.frame import GsFrame
from estimators.automl import H2OAutoML
import h2o


__all__ = ["lung_model"]


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def lung_model(train_info, feature, pred_info=None, leaderboard: list = None, d_output=None, prefix=None,
                    blending=None, weights_column=None, nthreads=10,  **kwargs):
    """"""

    # initiate h2o server
    logger.info(f"connect h2o server. <nthreads: {nthreads}; max_mem_size: {nthreads * 4 * 1000}M>")
    # cluster.init(nthreads=nthreads, max_mem_size=f"{nthreads * 4 * 1000}M")
    # connect to h2o server
    h2o.init(ip="localhost", port=12345, max_mem_size=f"{nthreads * 4 * 1000}M", nthreads=nthreads)
    
    # Generate data set
    logger.info(f"generate GsFrame...")
    gf_train = GsFrame(dataset_list=train_info, feature_list=feature)
    gf_pred = GsFrame(dataset_list=pred_info, feature_list=feature) if pred_info else None
    leaderboard_frame = GsFrame(dataset_list=leaderboard, feature_list=feature) if leaderboard else None
    blending_frame = GsFrame(feature_list=blending) if blending else None

    # Model Training
    logger.info(f"Model Training training...")
    model = H2OAutoML(**kwargs)
    model.train(d_output=d_output,
                prefix=prefix,
                x=gf_train.c_features,
                y="Response",
                training_frame=gf_train,
                predict_frame=gf_pred,
                leaderboard_frame=leaderboard_frame,
                blending_frame=blending_frame,
                weights_column=weights_column
                )
    # cluster.close()
    h2o.shutdown()
    logger.info(f"success!")
