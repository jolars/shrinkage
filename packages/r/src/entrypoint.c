#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

void R_init_shrinkage_extendr(DllInfo *dll);
void register_extendr_panic_hook(void);

/* Referencing the Rust registration function keeps the static library linked. */
void attribute_visible R_init_shrinkage(DllInfo *dll) {
    register_extendr_panic_hook();
    R_init_shrinkage_extendr(dll);
}
