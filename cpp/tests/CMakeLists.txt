add_executable(mathcore_tests test_symbolic.cpp)
target_link_libraries(mathcore_tests PRIVATE mathcore)
add_test(NAME mathcore_tests COMMAND mathcore_tests)

add_executable(test_errors test_errors.cpp)
target_link_libraries(test_errors PRIVATE mathcore)
add_test(NAME test_errors COMMAND test_errors)

add_executable(test_numeric test_numeric.cpp)
target_link_libraries(test_numeric PRIVATE mathcore)
add_test(NAME test_numeric COMMAND test_numeric)

add_executable(test_units test_units.cpp)
target_link_libraries(test_units PRIVATE mathcore)
add_test(NAME test_units COMMAND test_units)

add_executable(test_ode test_ode.cpp)
target_link_libraries(test_ode PRIVATE mathcore)
add_test(NAME test_ode COMMAND test_ode)
