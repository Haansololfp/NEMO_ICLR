import wandb
from celltype_ibl.models.metrics import topk


class wandblog:
    def __init__(self, name, config, entity, project, topk, wandb_dir):
        self.topk = topk
        self.name = name
        self.weights_artifact = wandb.Artifact(
            name="weights_" + self.name, type="weight_log"
        )
        self.cv_artifact = wandb.Artifact(
            name="lp_results_" + self.name, type="result_log"
        )

        self.run = wandb.init(
            project=project, config=config, entity=entity, name=name, dir=wandb_dir
        )

    def startdic(self, start=0):
        dic = {}
        for k in self.topk:
            dic["t" + str(k)] = start
        return dic

    def log(self, currlog):
        wandb.log(currlog)

    def log_results(self, cv_path, epoch):
        self.cv_artifact.add_file(local_path=cv_path)
        self.run.log_artifact(self.cv_artifact)
        self.cv_artifact = wandb.Artifact(
            name="lp_results_epoch_" + str(epoch) + self.name, type="result_log"
        )

    def log_weights(self, weight_path, k):
        self.weights_artifact = wandb.Artifact(
            name=f"weights_best_top_{k}" + self.name, type="weight_log"
        )
        self.weights_artifact.add_file(local_path=weight_path)
        self.run.log_artifact(self.weights_artifact)

    def log_weights_epoch(self, weight_path, epoch):
        self.weights_artifact = wandb.Artifact(
            name=f"weights_epoch_{epoch}" + self.name, type="weight_log"
        )
        self.weights_artifact.add_file(local_path=weight_path)
        self.run.log_artifact(self.weights_artifact)

    def end(self):
        wandb.finish()
