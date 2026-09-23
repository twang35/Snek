# The b45 reads page, rebuilt by hand

`viewer/pages/reads-b45.html` is self-contained: the batch's numbers are inlined. To refresh it (the
batch closing, a layout fix), pull the merged pass files off the desktop and run the reducer over them:

```
mkdir -p /tmp/b45 && rsync -a --exclude='*-s*of*.json' --include='*_checkpoint_evals_reads-*.json' --exclude='*' \
    the-claw-den:Snek/snek3/desktop/runs/ /tmp/b45/
cd snek3 && python plans/reads-b45/reduce_b45.py /tmp/b45 plans/reads-b45/reads-b45.template.html viewer/pages/reads-b45.html
```

then screenshot a view or two with headless Chrome, commit the page (viewer authorization), and deploy so the
desktop's next site build copies it. A one-off (`plans/quantile-reads.md` §5 step 9): nothing in `tools/` or
the site build knows about it, and that is deliberate.
