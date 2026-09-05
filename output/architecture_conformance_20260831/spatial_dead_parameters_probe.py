# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.6.0"]
# ///
# Run from repository root with D:/anaconda/python.exe -B <this-file>.
# Audit-only reproduction: no training, optimizer steps, or checkpoint IO.
import json,torch,sys
sys.path.insert(0,'.')
from models.mechanistic_retina.model import build_mechanistic_retina
from models.mechanistic_retina.contracts import MechanisticRetinaConfig,ArchitectureMode
from models.mechanistic_retina.pathway_spatial_geometry import PathwaySpatialGeometry
from training.mechanistic_retina.optimizer import build_phase1_optimizer
torch.set_num_threads(1); torch.manual_seed(831)
cp=torch.cartesian_prod(torch.arange(-4,5,dtype=torch.float64)*.04,torch.arange(-4,5,dtype=torch.float64)*.04)
cfg=MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE)
m=build_mechanistic_retina(cfg,cp,torch.zeros(4,2,dtype=torch.float64),('midget','midget','parasol','parasol'),('ON','OFF','ON','OFF')).eval()
x=torch.sin(torch.arange(1*32*81,dtype=torch.float64).reshape(1,32,81)*.017)
h=torch.zeros(1,32,4,dtype=torch.float64); h[:,1::5]=1
o=m.forward_sequence(x,observed_counts=h)
g=torch.autograd.grad(o.logits.square().sum(),m.shared_subunits.raw_connections)[0]
p=m.shared_subunits.raw_connections
with torch.no_grad(): p.add_(torch.arange(1,5,dtype=torch.float64))
n=m.forward_sequence(x,observed_counts=h)
opt=build_phase1_optimizer(m,learning_rate=.001)
single={'shape':list(p.shape),'requires_grad':p.requires_grad,'optimizer_count':sum(v is p for group in opt.param_groups for v in group['params']),'gradient':g.tolist(),'matrix':m.shared_subunits.connection_matrix().tolist(),'perturb_all_entries_logit_equal':torch.equal(o.logits,n.logits),'max_abs_logit_diff':float((o.logits-n.logits).abs().max())}
positions=torch.tensor([[0.,0.],[.04,0.],[.12,0.]],dtype=torch.float64)
geometry=PathwaySpatialGeometry(torch.ones(1,2,3,dtype=torch.float64),torch.tensor([[1.,0.,0.]],dtype=torch.float64),torch.tensor([[1.,0.,1.]],dtype=torch.float64))
irregular=build_mechanistic_retina(cfg,positions,positions[:1],('midget',),('ON',),pathway_spatial_geometry=geometry).eval()
ir=irregular.forward_sequence(torch.ones(1,32,3,dtype=torch.float64),observed_counts=torch.zeros(1,32,1,dtype=torch.float64))
print(json.dumps({'singleton_only_trainable':single,'irregular_geometry':{'accepted_and_forward_finite':bool(torch.isfinite(ir.logits).all()),'distance':positions[:,0].tolist(),'actual_BC':irregular.feature_bank.bc_support.tolist(),'actual_AC':irregular.feature_bank.ac_support.tolist(),'expected_full_BC':[[1.,1.,0.]],'expected_full_AC':[[1.,1.,1.]]}},indent=2))
