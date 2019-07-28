from fastai.imports import *
from fastai.vision import *

def one_batch_custom(datablock, batch_num=1,
                     ds_type:DatasetType=DatasetType.Train, detach:bool=True, denorm:bool=True, cpu:bool=True)->Collection[Tensor]:
    "Get one batch from the data loader of `ds_type`. Optionally `detach` and `denorm`."
    dl = datablock.dl(ds_type)
    w = datablock.num_workers
    datablock.num_workers = 0
    iterator = iter(dl)
    [next(iterator) for i in range(batch_num-1)]
    try:     x,y = next(iterator)
    finally: datablock.num_workers = w
    if detach: x,y = to_detach(x,cpu=cpu),to_detach(y,cpu=cpu)
    norm = getattr(datablock,'norm',False)
    if denorm and norm:
        x = datablock.denorm(x)
        if norm.keywords.get('do_y',False): y = self.denorm(y, do_x=True)
    return x,y


def show_batch_custom(self, batch_num=1, rows:int=5, ds_type:DatasetType=DatasetType.Train, reverse:bool=False, **kwargs)->None:
    """Show a batch of data , of batch number batch_num, in `ds_type` on a few `rows`.
    e.g show_batch_custom(data, batch_num=15)"""
    x,y = one_batch_custom(self,batch_num, ds_type, True, True)
    if reverse: x,y = x.flip(0),y.flip(0)
    n_items = rows **2 if self.train_ds.x._square_show else rows
    if self.dl(ds_type).batch_size < n_items: n_items = self.dl(ds_type).batch_size
    xs = [self.train_ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
    #TODO: get rid of has_arg if possible
    if has_arg(self.train_ds.y.reconstruct, 'x'):
        ys = [self.train_ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
    else : ys = [self.train_ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
    self.train_ds.x.show_xys(xs, ys, **kwargs)


def pred_batch_custom(self, ds_type:DatasetType=DatasetType.Valid, batch_num:int=1,  batch:Tuple=None, reconstruct:bool=False, with_dropout:bool=False) -> List[Tensor]:
    "Return output of the model on one batch from `ds_type` dataset."
    if batch is not None: xb,yb = batch
    else: xb,yb = one_batch_custom(self.data, batch_num=batch_num, ds_type=ds_type, detach=False, denorm=False)
    cb_handler = CallbackHandler(self.callbacks)
    xb,yb = cb_handler.on_batch_begin(xb,yb, train=False)
    with torch.no_grad():
        if not with_dropout: preds = loss_batch(self.model.eval(), xb, yb, cb_handler=cb_handler)
        else: preds = loss_batch(self.model.eval().apply(self.apply_dropout), xb, yb, cb_handler=cb_handler)
        res = _loss_func2activ(self.loss_func)(preds[0])
    if not reconstruct: return res
    res = res.detach().cpu()
    ds = self.dl(ds_type).dataset
    norm = getattr(self.data, 'norm', False)
    if norm and norm.keywords.get('do_y',False):
        res = self.data.denorm(res, do_x=True)
    return [ds.reconstruct(o) for o in res]



def show_results(self, ds_type=DatasetType.Valid, batch_num=1, rows:int=5, **kwargs):
    "Show `rows` result of predictions on `ds_type` dataset."
    #TODO: get read of has_arg x and split_kwargs_by_func if possible
    #TODO: simplify this and refactor with pred_batch(...reconstruct=True)
    n_items = rows ** 2 if self.data.train_ds.x._square_show_res else rows
    if self.dl(ds_type).batch_size < n_items: n_items = self.dl(ds_type).batch_size
    ds = self.dl(ds_type).dataset
    self.callbacks.append(RecordOnCPU())
    preds = pred_batch_custom(ds_type, batch_num=batch_num)
    *self.callbacks,rec_cpu = self.callbacks
    x,y = rec_cpu.input,rec_cpu.target
    norm = getattr(self.data,'norm',False)
    if norm:
        x = self.data.denorm(x)
        if norm.keywords.get('do_y',False):
            y     = self.data.denorm(y, do_x=True)
            preds = self.data.denorm(preds, do_x=True)
    analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
    preds = [ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs) for i in range(n_items)]
    xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
    if has_arg(ds.y.reconstruct, 'x'):
        ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
        zs = [ds.y.reconstruct(z, x=x) for z,x in zip(preds,xs)]
    else :
        ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
        zs = [ds.y.reconstruct(z) for z in preds]
    ds.x.show_xyzs(xs, ys, zs, **kwargs)