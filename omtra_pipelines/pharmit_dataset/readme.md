This pipeline processes data on jabba. We also periodically shuffle processed data from jabba back to masuda. 

To be able to shuttle things from jabba to masuda we have to setup a reverse tunnel. The command for doing so is:

```console
autossh -M 20000 -f -N -R 2222:localhost:22 jabba
```
