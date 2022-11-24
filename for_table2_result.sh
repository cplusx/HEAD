# 1. Train a resnet18 model to generate adversarial images, these adv images will be evaluated on cross model adversarial detection on a vgg16 model
python train_clean.py --dataset CIFAR10 --net resnet18 --out_dir ./clean_train/

declare -a Attacks=("PGD10" "BIM" "FGSM" "CW" "DeepFool" "OnePixel" "SparseFool" "AutoAttack" )

# 2. Compute adversarial images for resnet18
for atk in "${Attacks[@]}"; do
  echo "Generate adversarial images for $atk"
  python precompute_adversarial_examples.py --net resnet18 --adv_type $atk --split test --dataset CIFAR10
done

# 3. Compute Least Significant Component Feature
for atk in "${Attacks[@]}"; do
  echo "Compute LSCF features for $atk"
  python compute_LSCF_feature_cross_model.py --net resnet18 --adv_type $atk --split test --dataset CIFAR10 --feat_dim 32
done

# 4. Compute Hessian Feature
python precompute_multi_modulus_cross_model.py --net_of_adv resnet18  --net vgg16 --adv_type benign --split test --dataset CIFAR10 --modulus_mode l1
for atk in "${Attacks[@]}"; do
  echo "Compute HF features for $atk"
  python precompute_multi_modulus_cross_model.py --net_of_adv resnet18  --net vgg16 --adv_type $atk --split test --dataset CIFAR10 --modulus_mode l1
done

# 5. Result of KDE
python KDE_train_and_test_cross_model.py --adv_type overall --dataset CIFAR10 --kernel gaussian --bandwidth 4. --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
for atk in "${Attacks[@]}"; do
  echo "Evaluate $atk with KDE"
  python KDE_train_and_test_cross_model.py --adv_type $atk --dataset CIFAR10 --kernel gaussian --bandwidth 4. --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
done


# 6. Result of OCSVM
python OCSVM_train_and_test_cross_model.py --adv_type overall --dataset CIFAR10 --kernel rbf --nu 0.7 --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
for atk in "${Attacks[@]}"; do
  echo "Evaluate $atk with OCSVM"
  python OCSVM_train_and_test_cross_model.py --adv_type $atk --dataset CIFAR10 --kernel rbf --nu 0.7 --hessian_dim 13 --LSCF_dim 32 --norm_feat 1 --modulus_mode l1
done