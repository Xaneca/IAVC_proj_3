

Stage 3 treinavel, 3 mlps - deu overfit logo no inicio

época  1 | treino 1.649 | val 1.411 | MAE 5.01 anos, bins 57.2% | género 94.7% | expressão acc 62.2% F1 0.619 | 330s *
época  2 | treino 1.305 | val 1.383 | MAE 4.99 anos, bins 56.0% | género 93.9% | expressão acc 62.9% F1 0.624 | 592s *
época  3 | treino 1.133 | val 1.511 | MAE 5.09 anos, bins 54.0% | género 93.7% | expressão acc 61.0% F1 0.612 | 157s
época  4 | treino 0.969 | val 1.467 | MAE 4.86 anos, bins 57.2% | género 94.1% | expressão acc 62.0% F1 0.615 | 157s
época  5 | treino 0.812 | val 1.770 | MAE 4.83 anos, bins 58.2% | género 93.4% | expressão acc 60.7% F1 0.609 | 157s
época  6 | treino 0.658 | val 1.789 | MAE 4.94 anos, bins 56.8% | género 94.3% | expressão acc 61.5% F1 0.613 | 157s
época  7 | treino 0.532 | val 1.946 | MAE 4.91 anos, bins 56.3% | género 94.2% | expressão acc 61.5% F1 0.611 | 157s
Early stopping: 5 épocas sem melhorar.
Melhor loss de validação 1.383 -> models\multitask.pt

--------------------------------------------------------------------------

Stage 3 congelada, stage 4 treinavel - 3 mlps

UTKFace 16595 treino | AffectNet 30396 treino | 474 passos/época | cuda
Features do stage 4 | stages afinados [4] | congelados: stem + stages 1-3
época  1 | treino 1.615 | val 1.377 | MAE 4.74 anos, bins 57.3% | género 94.8% | expressão acc 61.5% F1 0.611 | 175s *
época  2 | treino 1.269 | val 1.339 | MAE 5.01 anos, bins 55.4% | género 95.1% | expressão acc 64.3% F1 0.642 | 145s *
época  3 | treino 1.107 | val 1.481 | MAE 4.79 anos, bins 58.5% | género 94.2% | expressão acc 61.3% F1 0.612 | 145s
época  4 | treino 0.928 | val 1.526 | MAE 4.69 anos, bins 58.8% | género 94.3% | expressão acc 63.2% F1 0.631 | 145s
época  5 | treino 0.734 | val 1.564 | MAE 4.71 anos, bins 57.3% | género 94.9% | expressão acc 62.7% F1 0.621 | 145s
época  6 | treino 0.565 | val 1.966 | MAE 4.74 anos, bins 59.8% | género 94.4% | expressão acc 61.3% F1 0.608 | 145s
época  7 | treino 0.438 | val 2.048 | MAE 4.89 anos, bins 58.0% | género 94.7% | expressão acc 60.2% F1 0.606 | 145s
Early stopping: 5 épocas sem melhorar.
Melhor loss de validação 1.339 -> models\multitask_s4.pt

--------------------------------------------------------------------------

Stage 3 congelado só treina mlp, sem stage 4

UTKFace 16595 treino | AffectNet 30396 treino | 474 passos/época | cuda
Features do stage 3 | stages afinados [] | congelados: stem + stages 1-3
época  1 | treino 1.910 | val 1.619 | MAE 5.96 anos, bins 49.5% | género 93.6% | expressão acc 57.7% F1 0.576 | 158s *
época  2 | treino 1.603 | val 1.520 | MAE 5.47 anos, bins 54.7% | género 93.9% | expressão acc 58.7% F1 0.583 | 125s *
época  3 | treino 1.526 | val 1.483 | MAE 5.39 anos, bins 54.6% | género 94.0% | expressão acc 60.2% F1 0.600 | 125s *
época  4 | treino 1.480 | val 1.454 | MAE 5.25 anos, bins 54.6% | género 94.0% | expressão acc 60.4% F1 0.601 | 125s *
época  5 | treino 1.440 | val 1.444 | MAE 5.27 anos, bins 54.0% | género 94.4% | expressão acc 61.2% F1 0.609 | 126s *
época  6 | treino 1.405 | val 1.443 | MAE 5.14 anos, bins 55.0% | género 94.6% | expressão acc 60.9% F1 0.605 | 126s *
época  7 | treino 1.380 | val 1.436 | MAE 5.35 anos, bins 53.4% | género 94.4% | expressão acc 62.5% F1 0.622 | 127s *
época  8 | treino 1.348 | val 1.444 | MAE 5.34 anos, bins 53.6% | género 94.9% | expressão acc 60.0% F1 0.596 | 126s
época  9 | treino 1.323 | val 1.417 | MAE 5.13 anos, bins 55.8% | género 94.9% | expressão acc 61.8% F1 0.615 | 126s *
época 10 | treino 1.297 | val 1.417 | MAE 5.10 anos, bins 56.2% | género 94.4% | expressão acc 62.0% F1 0.618 | 126s
época 11 | treino 1.270 | val 1.418 | MAE 5.06 anos, bins 57.2% | género 94.3% | expressão acc 61.7% F1 0.616 | 126s
época 12 | treino 1.239 | val 1.434 | MAE 5.06 anos, bins 55.9% | género 95.0% | expressão acc 61.6% F1 0.612 | 126s
época 13 | treino 1.217 | val 1.431 | MAE 5.20 anos, bins 54.2% | género 94.7% | expressão acc 61.8% F1 0.615 | 127s
época 14 | treino 1.188 | val 1.436 | MAE 5.07 anos, bins 55.5% | género 95.4% | expressão acc 61.5% F1 0.609 | 127s
Early stopping: 5 épocas sem melhorar.
Melhor loss de validação 1.417 -> models\multitask_frozen.pt

--------------------------------------------------

mesmo q o anterior, mas UTKFace com Black Border

Crops: UTKFace data/processed_black_border/utkface_112 | AffectNet data/processed/affectnet_112
utkface train: 281 sem crop em data\processed_black_border\utkface_112 (ignoradas)
utkface val: 54 sem crop em data\processed_black_border\utkface_112 (ignoradas)
UTKFace 16314 treino | AffectNet 30396 treino | 474 passos/época | cuda
Features do stage 3 | stages afinados [] | congelados: stem + stages 1-3
época  1 | treino 1.771 | val 1.538 | MAE 4.80 anos, bins 57.7% | género 94.1% | expressão acc 57.7% F1 0.576 | 170s *
época  2 | treino 1.506 | val 1.469 | MAE 4.73 anos, bins 57.8% | género 94.3% | expressão acc 58.7% F1 0.583 | 133s *
época  3 | treino 1.435 | val 1.437 | MAE 4.81 anos, bins 56.6% | género 94.9% | expressão acc 60.2% F1 0.600 | 134s *
época  4 | treino 1.387 | val 1.425 | MAE 4.82 anos, bins 56.9% | género 94.3% | expressão acc 60.4% F1 0.601 | 133s *
época  5 | treino 1.347 | val 1.405 | MAE 4.65 anos, bins 59.2% | género 95.1% | expressão acc 61.2% F1 0.609 | 131s *
época  6 | treino 1.308 | val 1.413 | MAE 4.57 anos, bins 60.0% | género 95.0% | expressão acc 60.9% F1 0.605 | 129s
época  7 | treino 1.277 | val 1.403 | MAE 4.80 anos, bins 55.5% | género 94.9% | expressão acc 61.4% F1 0.611 | 128s *
época  8 | treino 1.247 | val 1.389 | MAE 4.62 anos, bins 59.3% | género 94.8% | expressão acc 61.8% F1 0.614 | 128s *
época  9 | treino 1.214 | val 1.405 | MAE 4.64 anos, bins 57.8% | género 94.8% | expressão acc 61.7% F1 0.614 | 128s
época 10 | treino 1.183 | val 1.421 | MAE 4.62 anos, bins 58.3% | género 94.6% | expressão acc 61.4% F1 0.612 | 128s
época 11 | treino 1.153 | val 1.405 | MAE 4.54 anos, bins 59.5% | género 95.0% | expressão acc 61.9% F1 0.619 | 128s
época 12 | treino 1.122 | val 1.431 | MAE 4.55 anos, bins 58.8% | género 94.9% | expressão acc 61.1% F1 0.607 | 128s
época 13 | treino 1.093 | val 1.422 | MAE 4.52 anos, bins 59.3% | género 95.1% | expressão acc 61.8% F1 0.616 | 128s
Early stopping: 5 épocas sem melhorar.
Melhor loss de validação 1.389 -> models\multitask_frozen.pt