import React, { Suspense } from 'react'
import {
    Route,
    createBrowserRouter,
    createRoutesFromElements,
    RouterProvider
} from 'react-router-dom';
import AppUI from './components/AppUI';
import LoadingPage from './Loading';
import Home from './Home';
import App from './App';
import Instructions from './Instructions';

const Router = () => {
    let router = createBrowserRouter(
        createRoutesFromElements(
            <Route path='/' element={<AppUI />} >
                <Route index element={<Home />} />
                <Route path='app' element={<App />} />
                <Route path='instructions' element={<Instructions />} />
            </Route>
        )
    )
    return (
        <Suspense fallback={<LoadingPage />}>
            <RouterProvider router={router} />
        </Suspense>
    )
}

export default Router