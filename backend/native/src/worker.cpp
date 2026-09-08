#include "simulation.hpp"
#include <dlfcn.h>
#include <iostream>
#include <sys/resource.h>
int main(int argc, char **argv) {
  using namespace sa;
  if (argc != 3)
    return 2;
  rlimit cpu{8, 8}, file{64 * 1024 * 1024, 64 * 1024 * 1024};
  setrlimit(RLIMIT_CPU, &cpu);
  setrlimit(RLIMIT_FSIZE, &file);
  J response;
  void *module = nullptr;
  void (*close_python)() = nullptr;
  try {
    J input = J::parse(read(argv[1]));
    Callback callback = nullptr;
    PortfolioCallback portfolio_callback = nullptr;
    bool multi = input.contains("assets");
    if (input["language"] == "cpp") {
      module = dlopen(input["module_path"].get<std::string>().c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!module)
        throw Error(422, dlerror());
      if (multi)
        portfolio_callback = reinterpret_cast<PortfolioCallback>(dlsym(module, "on_portfolio"));
      else
        callback = reinterpret_cast<Callback>(dlsym(module, "on_bar"));
      if (!(multi ? bool(portfolio_callback) : bool(callback)))
        throw Error(422, multi ? "Multi-stock C++ strategies must export on_portfolio."
                               : "C++ strategy must export on_bar.");
    } else {
      auto adapter = fs::absolute(argv[0]).parent_path() / "stockassist-python-adapter.so";
      module = dlopen(adapter.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!module)
        throw Error(422, dlerror());
      using Load = Callback (*)(const char *, const char *, const char *);
      auto load = reinterpret_cast<Load>(dlsym(module, "sa_python_load"));
      close_python = reinterpret_cast<void (*)()>(dlsym(module, "sa_python_close"));
      if (!load || !close_python)
        throw Error(422, "Python adapter exports unavailable.");
      if (multi) {
        using MultiLoad = PortfolioCallback (*)(const char *, const char *, const char *);
        auto multi_load = reinterpret_cast<MultiLoad>(dlsym(module, "sa_python_load_portfolio"));
        if (!multi_load)
          throw Error(422, "Portfolio Python adapter is unavailable.");
        portfolio_callback =
            multi_load(input["module_path"].get<std::string>().c_str(),
                       input["assets"].dump().c_str(), input["params"].dump().c_str());
      } else
        callback = load(input["module_path"].get<std::string>().c_str(),
                        input["bars"].dump().c_str(), input["params"].dump().c_str());
      if (!(multi ? bool(portfolio_callback) : bool(callback)))
        throw Error(422, "Python strategy could not be loaded. See worker logs.");
    }
    J result = multi ? simulate_portfolio(input["assets"], portfolio_callback, input["params"],
                                          input["settings"])
                     : simulate(input["bars"], callback, input["params"], input["settings"]);
    add_comparison(result, input, input["settings"]);
    response = {{"result", result}};
  } catch (const std::exception &e) {
    response = {{"error", e.what()}};
  }
  try {
    write(argv[2], response.dump());
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  if (close_python)
    close_python();
  if (module)
    dlclose(module);
  return 0;
}
