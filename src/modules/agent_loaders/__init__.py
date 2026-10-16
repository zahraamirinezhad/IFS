REGISTRY = {}

from .rnn_eval_agent_loader import RNNEvalAgentLoader
from .rnn_train_agent_loader import RNNTrainAgentLoader
from .poam_eval_agent_loader import POAMEvalAgentLoader
from .poam_train_agent_loader import POAMTrainAgentLoader
from .ifs_train_agent_loader import IFSTrainAgentLoader
from .ifs_eval_agent_loader import IFSEvalAgentLoader


REGISTRY["rnn_eval_agent_loader"] = RNNEvalAgentLoader
REGISTRY["rnn_train_agent_loader"] = RNNTrainAgentLoader
REGISTRY["poam_eval_agent_loader"] = POAMEvalAgentLoader
REGISTRY["poam_train_agent_loader"] = POAMTrainAgentLoader
REGISTRY["ifs_train_agent_loader"] = IFSTrainAgentLoader
REGISTRY["ifs_eval_agent_loader"] = IFSEvalAgentLoader
