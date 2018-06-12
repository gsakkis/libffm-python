.PHONY: libffm clean

libffm:
	$(MAKE) -C libffm

clean:
	$(MAKE) -C libffm clean
	rm -rf build/ dist/ ffm.egg-info/ ffm/__pycache__/
