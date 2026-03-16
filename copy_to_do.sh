#!/bin/bash
# always copy the database before pulling it
#scp toxpro@192.241.131.84:/home/toxpro/instance/toxpro.sqlite instance/
scp -r ./* toxpro@192.241.131.84:/home/toxpro