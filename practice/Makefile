default: ad_template.ipynb


.PHONY: phony_explicit
phony_explicit:

%_template.ipynb: %_solution.ipynb phony_explicit
	nbstripout $<
	rm $@
	python -m paderbox.utils.strip_solution --old-path $< --new-path $@

