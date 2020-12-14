# Eval object 1-20
for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
do
    ae_embed cad_autoencoder/obj${OBJ_ID}
    wait
done

wait
