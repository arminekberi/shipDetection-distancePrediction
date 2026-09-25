"""Fine-tune a reviewed snapshot; persist progress and automatically compare held-out results."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback


def write_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='mps')
    args=p.parse_args()
    args.out=args.out.resolve()
    if (args.out/'status.json').exists():
        raise FileExistsError('This output already has a training job; choose a new --out directory')
    args.out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR',str(args.out/'ultralytics_config'))
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('WANDB_MODE','disabled')
    from ultralytics import YOLO, settings
    import torch
    from viewer.dataset_review import digest
    # Use only local logging for this job.
    settings.update({'sync':False,'wandb':False,'clearml':False,'comet':False,'mlflow':False})
    torch.set_num_threads(6)
    status={'status':'starting','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
            'data':str(args.data.resolve()),'source_weights':str(args.weights.resolve()),
            'source_weights_sha256':digest(args.weights),'device':args.device,'epochs_requested':args.epochs,
            'imgsz':args.imgsz,'batch':args.batch}
    status_path=args.out/'status.json'
    write_json(status_path,status)

    def progress(trainer):
        status.update(status='training',epoch=trainer.epoch+1,
                      metrics={k:float(v) for k,v in trainer.metrics.items()},
                      checkpoint_dir=str(trainer.wdir),updated_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path,status)
        print('EPOCH_STATUS '+json.dumps(status),flush=True)

    try:
        model=YOLO(str(args.weights.resolve()))
        model.add_callback('on_fit_epoch_end',progress)
        model.train(data=str(args.data.resolve()),project=str(args.out),name='finetune',exist_ok=False,
                    epochs=args.epochs,patience=10,imgsz=args.imgsz,batch=args.batch,device=args.device,
                    workers=0,cache=False,optimizer='AdamW',lr0=.0002,lrf=.05,weight_decay=.0005,
                    warmup_epochs=2,cos_lr=True,close_mosaic=5,mosaic=.5,scale=.3,
                    degrees=3,flipud=0,fliplr=.5,mixup=0,seed=17,deterministic=True,
                    amp=False,plots=True,save_period=5,verbose=True)
        best=Path(model.trainer.best)
        status.update(status='evaluating',best_checkpoint=str(best),best_sha256=digest(best))
        write_json(status_path,status)
        comparison={}
        for label,weights in [('baseline',args.weights),('finetuned',best)]:
            candidate=YOLO(str(weights))
            comparison[label]={}
            for split in ('val','test'):
                metrics=candidate.val(data=str(args.data.resolve()),split=split,imgsz=args.imgsz,
                                      batch=args.batch,device=args.device,workers=0,plots=True,
                                      project=str(args.out/'evaluation'),name=f'{label}_{split}')
                comparison[label][split]={k:float(v) for k,v in metrics.results_dict.items()}
                write_json(args.out/'comparison.json',comparison)
        status.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),comparison=comparison)
        write_json(status_path,status)
    except BaseException as exc:
        status.update(status='failed',error=str(exc),traceback=traceback.format_exc(),
                      updated_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path,status)
        raise


if __name__ == '__main__':
    main()
