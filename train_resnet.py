# train_vgg.py
import os, csv, json, time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime

import os

from ObjectDetectionProblemFromScratch.voc_dataloader import get_train_dataloader, get_test_dataloader
from ObjectDetectionProblemFromScratch.multibox_loss import MultiBoxLoss

from res_net_18 import SSD_ResNet18
import config_resnet as C

#логирование
from log_setup import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)



def main():
    logger.info(f"OVERLAP_THRESHOLD: {C.OVERLAP_THRESHOLD}")
    logger.info(f"NEG_POS_RATIO: {C.NEG_POS_RATIO}")
    logger.info(f"VARIANCE: {C.VARIANCE}")
    logger.info(f"Размер батча : {C.BATCH_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Start. Device: {device}")
    

    # --- Data ---
    train_ann = os.path.join(C.VOC_ROOT, "ImageSets/Main/trainval.txt")
    test_ann  = os.path.join(C.VOC_ROOT, "ImageSets/Main/test.txt")

    train_loader = get_train_dataloader(C.VOC_ROOT, train_ann, C.BATCH_SIZE, C.NUM_WORKERS)
    logger.info(f"Размер train датасета : {len(train_loader.dataset)}")

    test_loader  = get_test_dataloader (C.VOC_ROOT, test_ann , C.BATCH_SIZE, C.NUM_WORKERS)
    logger.info(f"Размер test датасета : {len(test_loader.dataset)}")


    # --- Model ---
    model = SSD_ResNet18(C.NUM_CLASSES, C.PRIOR_CFG).to(device)
    criterion = MultiBoxLoss(
         overlap_threshold=C.OVERLAP_THRESHOLD,
         neg_pos_ratio=C.NEG_POS_RATIO,
         variance=C.VARIANCE,
    )

    optimizer = optim.SGD(
        model.parameters(), 
        lr=C.LR,
        momentum=C.MOMENTUM, 
        weight_decay=C.WEIGHT_DECAY,
    )
    scheduler = MultiStepLR(optimizer, milestones=C.LR_STEPS, gamma=C.LR_GAMMA)

    # --- Создаём папку для запуска ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"ssd_vgg16_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # --- Сохраняем конфиг ---
    config_dump = {
        "num_classes": C.NUM_CLASSES,
        "batch_size": C.BATCH_SIZE,
        "epochs": C.EPOCHS,
        "lr": C.LR,
        "momentum": C.MOMENTUM,
        "weight_decay": C.WEIGHT_DECAY,
        "lr_steps": C.LR_STEPS,
        "lr_gamma": C.LR_GAMMA,
        "overlap_threshold": C.OVERLAP_THRESHOLD,
        "neg_pos_ratio": C.NEG_POS_RATIO,
        "variance": C.VARIANCE,
        "prior_cfg": C.PRIOR_CFG,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2, default=str)

    # --- Сохраняем архитектуру ---
    with open(os.path.join(run_dir, "arch.txt"), "w") as f:
        f.write(str(model))

    # --- CSV-лог по эпохам ---
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "loss_l", "loss_c", "loss_total", "time_sec"])

    best_loss = float("inf")

    train_losses = []
    val_losses   = []

    for epoch in range(1, C.EPOCHS + 1):
        logger.info(f"Epoch: {epoch}")

        model.train()
        sum_l = 0.0 # Сумарная ошибка локализации
        sum_c = 0.0 # Cумарная ошибка классификации
        sum_t = 0.0 # Сумарная общая ошибка
        n_batches = 0

        # обучение
        for batch_idx, batch in enumerate(train_loader):
            images, box_ss, label_ss = batch
            if images.numel() == 0:
                 continue
            logger.debug(f"Image shape: {images.shape}")
            
            images = images.to(device)

            gt_box_s   = [box.float().to(device) for box in box_ss]
            gt_label_s = [label.long().to(device)  for label in label_ss]
            targets = (gt_label_s, gt_box_s)

            loc_pred, conf_pred, priors = model(images)
            logger.debug(priors.shape)
            loss_l, loss_c = criterion((loc_pred, conf_pred, priors), targets)
            loss = loss_l + loss_c

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()


            logger.debug(
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"loss_l: {loss_l.item():.4f} | "
                f"loss_c: {loss_c.item():.4f} | "
                f"loss_total: {loss.item():.4f}"
            )


            sum_l += loss_l.item()
            sum_c += loss_c.item()
            sum_t += loss.item()
            n_batches += 1

        avg_l = sum_l / n_batches
        avg_c = sum_c / n_batches
        avg_t = sum_t / n_batches
        train_losses.append(avg_t)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
                f"Epoch {epoch} finished | "
                f"avg_loss_l: {avg_l:.4f} | "
                f"avg_loss_c: {avg_c:.4f} | "
                f"avg_loss_total: {avg_t:.4f} | "
                f"lr: {current_lr:.6f} | "
                f"batches: {n_batches}"
            )
        
        scheduler.step()

        # Валидация
        model.eval()
        val_sum_l = 0
        val_sum_c = 0
        val_sum_t = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                images, box_ss, label_ss = batch
                if images.numel() == 0:
                    continue

                images = images.to(device)

                gt_box_s   = [box.float().to(device) for box in box_ss]
                gt_label_s = [label.long().to(device)  for label in label_ss]
                targets = (gt_label_s, gt_box_s)

                loc_pred, conf_pred, priors = model(images)
                val_loss_l, val_loss_c = criterion((loc_pred, conf_pred, priors), targets)
                val_loss = val_loss_l + val_loss_c

                val_sum_l += val_loss_l.item()
                val_sum_c += val_loss_c.item()
                val_sum_t += val_loss.item()
                val_batches += 1

        val_avg_l = val_sum_l / val_batches
        val_avg_c = val_sum_c / val_batches
        val_avg_t = val_sum_t / val_batches

        logger.info(
            f"Epoch {epoch} VALIDATION | "
            f"val_loss_l: {val_avg_l:.4f} | "
            f"val_loss_c: {val_avg_c:.4f} | "
            f"val_loss_total: {val_avg_t:.4f} | "
            f"batches: {val_batches}"
        )

        val_losses.append(val_avg_t)
        
if __name__ == "__main__":
    main()